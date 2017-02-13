#include "../include/place_recognition.h"

////////////////
//// *** RUN ***
////////////////
//// Constructor
// Pre: nh.ok() == true
// Post: --Calls 'loadCodebook' --Calls 'subscribeToImages'
FABMapRun::FABMapRun(ros::NodeHandle nh) : OpenFABMap2(nh), sizeImgSeq(0)
{
    // Load trained data
    bool goodLoad = loadCodebook();

    if (goodLoad)
    {
        ROS_INFO("--Codebook successfully loaded!--");
        // Read private parameters
        ros::NodeHandle local_nh_("~");
        local_nh_.param<int>("maxMatches", maxMatches_, 0);
        local_nh_.param<double>("minMatchValue", minMatchValue_, 0.0);
        local_nh_.param<bool>("DisableSelfMatch", disable_self_match_, false);
        local_nh_.param<int>("SelfMatchWindow", self_match_window_, 1);
        local_nh_.param<bool>("DisableUnknownMatch", disable_unknown_match_, false);
        local_nh_.param<bool>("AddOnlyNewPlaces", only_new_places_, false);

        // Setup publisher
        pub_ = nh_.advertise<openfabmap2_msgs::Match>("appearance_matches", 1000);

        // Initialise for the first to contain
        // - Match to current
        // - Match to nothing
        confusionMat = cv::Mat::zeros(2,2,CV_64FC1);

        // Set callback
        subscribeToImages();
    }
    else
    {
        shutdown();
    }
}

//// Destructor
FABMapRun::~FABMapRun()
{
}

//// Image Callback
// Pre: image_msg->encoding == end::MONO8
// Post: -Matches to 'image_msg' published on pub_
//		 -'firstFrame_' blocks initial nonsensical self match case
void FABMapRun::processImgCallback(const sensor_msgs::ImageConstPtr& image_msg)
{
    ROS_DEBUG_STREAM("OpenFABMap2-> Processing image sequence number: " << image_msg->header.seq);

    cv_bridge::CvImagePtr cv_ptr;

    //Added
    sensor_msgs::Image image_prova;
    image_prova = *image_msg; 
    image_prova.header.seq+=sizeImgSeq; 
    
    try
    {
        // TODO: toCvShare should be used for 'FABMapRun'
        cv_ptr = cv_bridge::toCvCopy(image_prova, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    ROS_DEBUG("Received %d by %d image, depth %d, channels %d",
        cv_ptr->image.cols,
        cv_ptr->image.rows,
        cv_ptr->image.depth(),
        cv_ptr->image.channels());

    cv::Mat bow;
    ROS_DEBUG("Detector.....");
    detector->detect(cv_ptr->image, kpts);
    ROS_DEBUG("Compute descriptors...");
    bide->compute(cv_ptr->image, kpts, bow);

    // Check if the frame could be described
    if (!bow.empty() && kpts.size() > minDescriptorCount_)
    {

        // IF is NOT the first frame processed
        if (!firstFrame_)
        {
            ROS_DEBUG("Compare bag of words...");
            std::vector<cv::of2::IMatch> matches;

            // Find match likelyhoods for this 'bow'
            fabMap->compare(bow, matches, !only_new_places_);

            // Sort matches with oveloaded '<' into
            // Accending 'match' order
            std::sort(matches.begin(), matches.end());

            // Add BOW
            if (only_new_places_)
            {
                // Check if fabMap believes this to be a new place
                if (matches.back().imgIdx == -1)
                {
                    ROS_WARN_STREAM("Adding bow of new place...");
                    fabMap->add(bow);

                    // store the mapping from 'seq' to match ID
                    toImgSeq.push_back(image_msg->header.seq);
                }
            }
            else
            {
                // store the mapping from 'seq' to match ID
                toImgSeq.push_back(image_msg->header.seq);
            }

            // Build message
            openfabmap2_msgs::Match matched;
            matched.fromImgSeq = image_msg->header.seq;

            // IMAGE seq number
            int matchImgSeq;

            // Prepare message in Decending match likelihood order
            for (std::vector<cv::of2::IMatch>::reverse_iterator matchIter = matches.rbegin();
                matchIter != matches.rend();
                ++matchIter)
            {
                // Limit the number of matches published (by 'maxMatches_' OR 'minMatchValue_')
                if ((matched.toImgSeq.size() == maxMatches_ && maxMatches_ != 0)
                    || matchIter->match < minMatchValue_)
                {
                    break;
                }

                ROS_DEBUG_STREAM(
                    "QueryIdx " << matchIter->queryIdx <<
                    " ImgIdx " << matchIter->imgIdx <<
                    " Likelihood " << matchIter->likelihood <<
                    " Match " << matchIter->match);

                // ROS_INFO_STREAM("Primo elemento DI toImgSeq--->>>   " << toImgSeq.at(0));
                // ROS_INFO_STREAM("DIMENSIONE DI toImgSeq--->>>   " << toImgSeq.size());
                // ROS_INFO_STREAM("INDICE A CUI DEVO ACCEDERE--->>>   " << matchIter->imgIdx);

                // Lookup IMAGE seq number from MATCH seq number
                matchImgSeq = matchIter->imgIdx > -1 ? toImgSeq.at(matchIter->imgIdx) : -1;

                // Additionally if required,
                // --do NOT return matches below self matches OR new places ('-1')
                if ((matchImgSeq >= matched.fromImgSeq-self_match_window_ && disable_self_match_)
                    || (matchImgSeq == -1 && disable_unknown_match_))
                {
                    break;
                }

                // Add the Image seq number and its match likelihood
                matched.toImgSeq.push_back(matchImgSeq);
                matched.toImgMatch.push_back(matchIter->match);
            }

            // IF filtered matches were found
            if (matched.toImgSeq.size() > 0)
            {
                // Publish current matches
                pub_.publish(matched);

                if (visualise_)
                {
                    visualiseMatches(matches);
                }
            }
        }
        else
        {
            // First frame processed
            fabMap->add(bow);

            // store the mapping from 'seq' to match ID
            toImgSeq.push_back(image_msg->header.seq);

            firstFrame_ = false;
        }
    }
    else
    {
        ROS_WARN("--Image not descriptive enough, ignoring."); 
    }
}

//// Visualise Matches
// Pre:
// Post:
void FABMapRun::visualiseMatches(std::vector<cv::of2::IMatch> &matches)
{

    int numMatches = matches.size();

    cv::Mat newConfu = cv::Mat::zeros(numMatches, numMatches, CV_64FC1);
    ROS_DEBUG_STREAM("'newConfu -> rows: " << newConfu.rows
        << " cols: " << newConfu.cols);
    cv::Mat roi(newConfu, cv::Rect(0, 0, confusionMat.cols, confusionMat.rows));
    ROS_DEBUG_STREAM("'ROI -> rows: " << roi.rows
        << " cols: " << roi.cols);
    confusionMat.copyTo(roi);

    for (std::vector<cv::of2::IMatch>::reverse_iterator matchIter = matches.rbegin();
        matchIter != matches.rend();
        ++matchIter)
    {
        // Skip null match
        if (matchIter->imgIdx == -1)
        {
            continue;
        }

        ROS_DEBUG_STREAM("QueryIdx " << matchIter->queryIdx <<
            " ImgIdx " << matchIter->imgIdx <<
            " Likelihood " << matchIter->likelihood <<
            " Match " << matchIter->match);

        ROS_DEBUG_STREAM("--About to multi " << 255 << " by " << (double)matchIter->match);
        ROS_DEBUG_STREAM("---Result " << floor(255*((double)matchIter->match)));
        newConfu.at<double>(numMatches-1, matchIter->imgIdx) = 255*(double)matchIter->match;
        ROS_DEBUG_STREAM("-Uchar: " << newConfu.at<double>(numMatches-1, matchIter->imgIdx)
            << " at (" << numMatches << ", " << matchIter->imgIdx << ")");
    }
    newConfu.at<double>(numMatches-1, numMatches-1) = 255.0;
    ROS_DEBUG_STREAM("-Value: " << newConfu.at<double>(numMatches-1,numMatches-1)
        << " at (" << numMatches << ", " << numMatches << ")");

    confusionMat = newConfu.clone();
    ROS_DEBUG_STREAM("'confusionMat -> rows: " << confusionMat.rows
        << " cols: " << confusionMat.cols);

    // added: da sistemare
	//cv::Mat newConfuSmall(newConfu.rows, newConfu.cols, CV_64FC1);
	//resize(newConfuSmall, newConfu, cv::Size(500, 500), 0, 0, CV_INTER_NN);
	//cv::imshow("Confusion Matrix", newConfu);

    cv::imshow("Confusion Matrix", newConfu);
    cv::waitKey(10);
}

//// File loader
// Pre:
// Post:
bool FABMapRun::loadCodebook()
{
    ROS_INFO("Loading codebook...");

    cv::FileStorage fs;

    fs.open(vocabPath_, cv::FileStorage::READ);
    fs["Vocabulary"] >> vocab;
    fs.release();
    ROS_INFO("Vocabulary with %d words, %d dims loaded",vocab.rows,vocab.cols);

    fs.open(clTreePath_, cv::FileStorage::READ);
    fs["Tree"] >> clTree;
    fs.release();
    ROS_INFO("Chow Liu Tree loaded");

    fs.open(trainbowsPath_, cv::FileStorage::READ);
    fs["Trainbows"] >> trainbows;
    fs.release();
    ROS_INFO("Trainbows loaded");

    ROS_INFO("Setting the Vocabulary...");
    bide->setVocabulary(vocab);

    ROS_INFO("Initialising FabMap2 with Chow Liu tree...");

    // added
    ROS_INFO("Load BOWS from storage file");
    fs.open("/home/serl/STAR/src/localization/openfabmap2/codebooks/new/storage.yml", cv::FileStorage::READ);
    fs["Storage"] >> bows;
    fs.release();

    //added
    ROS_INFO("Load header image from image file");
    fs.open("/home/serl/STAR/src/localization/openfabmap2/codebooks/new/image.yml", cv::FileStorage::READ);
    fs["Image"] >> toImgSeq;
    fs.release();


    // Get additional parameters
    ros::NodeHandle local_nh_("~");

    //create options flags
    std::string new_place_method, bayes_method;
    int simple_motion;
    local_nh_.param<std::string>("NewPlaceMethod", new_place_method, "Meanfield");
    local_nh_.param<std::string>("BayesMethod", bayes_method, "ChowLiu");
    local_nh_.param<int>("SimpleMotion", simple_motion, 0);

    int options = 0;
    if (new_place_method == "Sampled")
        options |= cv::of2::FabMap::SAMPLED;
    else
        options |= cv::of2::FabMap::MEAN_FIELD;

    if (bayes_method == "ChowLiu")
        options |= cv::of2::FabMap::CHOW_LIU;
    else
        options |= cv::of2::FabMap::NAIVE_BAYES;

    if (simple_motion)
        options |= cv::of2::FabMap::MOTION_MODEL;

    //create an instance of the desired type of FabMap
    std::string fabMapVersion;
    double pzge, pzgne;
    int num_samples;
    local_nh_.param<std::string>("FabMapVersion", fabMapVersion, "FABMAPFBO");
    local_nh_.param<double>("PzGe", pzge, 0.39);
    local_nh_.param<double>("PzGne", pzgne, 0);
    local_nh_.param<int>("NumSamples", num_samples, 3000);

    if(fabMapVersion == "FABMAP1")
    {
        fabMap = new cv::of2::FabMap1(clTree, pzge, pzgne, options, num_samples);
    }
    else if (fabMapVersion == "FABMAPLUT")
    {
        int lut_precision;
        local_nh_.param<int>("Precision", lut_precision, 6);
        fabMap = new cv::of2::FabMapLUT(clTree, pzge, pzgne, options, num_samples,
            lut_precision);
    }
    else if (fabMapVersion == "FABMAPFBO")
    {
        double fbo_rejection_threshold, fbo_psgd;
        int fbo_bisection_start, fbo_bisection_its;
        local_nh_.param<double>("RejectionThreshold", fbo_rejection_threshold, 1e-6);
        local_nh_.param<double>("PsGd", fbo_psgd, 1e-6);
        local_nh_.param<int>("BisectionStart", fbo_bisection_start, 512);
        local_nh_.param<int>("BisectionIts", fbo_bisection_its, 9);
        fabMap = new cv::of2::FabMapFBO(clTree, pzge, pzgne, options, num_samples,
            fbo_rejection_threshold, fbo_psgd, fbo_bisection_start, fbo_bisection_its);

    }
    else if (fabMapVersion == "FABMAP2")
    {
        fabMap = new cv::of2::FabMap2(clTree, pzge, pzgne, options);
    }
    else
    {
        ROS_ERROR("Could not identify openFABMAPVersion from node params");
        return false;
    }

    ROS_INFO("Adding the trained bag of words...");
    fabMap->addTraining(trainbows);

    // added
    ROS_INFO("Caricamento delle BOW da Storage");
    for (int i = 0; i < bows.rows; i++) {
     fabMap->add(bows.row(i));
 }

 sizeImgSeq=toImgSeq.size();

 return true;
}

//// Unlink Callback
// Pre:
// Post: --Cleanup
void FABMapRun::shutdown()
{
    // Flag this worker as complete
    working_ = false;

    if (sub_.getNumPublishers() > 0)
    {
        ROS_WARN_STREAM("Shutting down " << sub_.getNumPublishers() << " subscriptions...");
        sub_.shutdown();
        nh_.shutdown();
    }
    else
    {
        ROS_ERROR_STREAM("-> " << sub_.getNumPublishers() << " subscriptions when shutting down..");
    }
}
// end class implementation FABMapRun
