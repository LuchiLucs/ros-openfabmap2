#include "../include/place_exploration.h"

////////////////
    //// *** PRERUN ***
    ////////////////
    //// Constructor
    // Pre: nh.ok() == true
    // Post: --Calls 'loadCodebook' --Calls 'subscribeToImages'
    //Aggiunta da noi tutta la classe
    FABMapPreRun::FABMapPreRun(ros::NodeHandle nh) : OpenFABMap2(nh), totalBOW(0)
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
            local_nh_.param<int>("maxBOW", maxBOW_, 100);
            local_nh_.param<std::string>("storagePath", storagePath_, "storage.yml");
            local_nh_.param<std::string>("imagePath", imagePath_, "image.yml");

            // Set callback
            subscribeToImages();
        }
        else
        {
            shutdown();
        }
    }

    //// Destructor
    FABMapPreRun::~FABMapPreRun()
    {
    }


    //// File loader
    // Pre:
    // Post:
    bool FABMapPreRun::loadCodebook()
    {
        ROS_INFO("Loading codebook...");

        cv::FileStorage fs;

        fs.open(vocabPath_,
                        cv::FileStorage::READ);
        fs["Vocabulary"] >> vocab;
        fs.release();
        ROS_INFO("Vocabulary with %d words, %d dims loaded",vocab.rows,vocab.cols);

        fs.open(clTreePath_,
                        cv::FileStorage::READ);
        fs["Tree"] >> clTree;
        fs.release();
        ROS_INFO("Chow Liu Tree loaded");

        fs.open(trainbowsPath_,
                        cv::FileStorage::READ);
        fs["Trainbows"] >> trainbows;
        fs.release();
        ROS_INFO("Trainbows loaded");

        ROS_INFO("Setting the Vocabulary...");
        bide->setVocabulary(vocab);

        ROS_INFO("Initialising FabMap2 with Chow Liu tree...");

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

        return true;
    }


    void FABMapPreRun::shutdown()
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

    //// Image Callback
    // Pre: image_msg->encoding == end::MONO8
    // Post: -Matches to 'image_msg' published on pub_
    //           -'firstFrame_' blocks initial nonsensical self match case
    void FABMapPreRun::processImgCallback(const sensor_msgs::ImageConstPtr& image_msg)
    {
        ROS_INFO_STREAM("Processando immagine... " << image_msg->header.seq);
        ROS_DEBUG_STREAM("OpenFABMap2-> Processing image sequence number: " << image_msg->header.seq);

        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            // TODO: toCvShare should be used for 'FABMapRun'
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
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
                std::vector<cv::of2::IMatch> matches;   // definito nel file openfabmap.hpp

                // Find match likelyhoods for this 'bow'
                fabMap->compare(bow,matches,!only_new_places_);

                // Sort matches with oveloaded '<' into
                // Accending 'match' order
                std::sort(matches.begin(), matches.end());

                // Add BOW
                if (only_new_places_)
                {
                    // Check if fabMap believes this to be a new place
                    if (matches.back().imgIdx == -1)
                    {
                        //AGGIUNTO DA GIOVANNI
                        ROS_INFO("Push BOW in BOWS");
                        bows.push_back(bow);
                        ROS_INFO_STREAM("Immagine salvata con header " << image_msg->header.seq);
                        toImgSeq.push_back(image_msg->header.seq);

                        totalBOW++;
                        ROS_INFO_STREAM("totalBOW = " << totalBOW);
                    }
                    else{
                        ROS_INFO("NON e' un nuovo posto, non faccio nulla");
                    }
                }
                else
                {
                    ROS_INFO("Push BOW in BOWS");
                    bows.push_back(bow);
                    ROS_INFO_STREAM("Immagine salvata con header " << image_msg->header.seq);
                    toImgSeq.push_back(image_msg->header.seq);

                    totalBOW++;
                    ROS_INFO_STREAM("totalBOW = " << totalBOW);
                }
            }
            else
            {   //first frame processing
                //AGGIUNTO DA GIOVANNI
                ROS_INFO("Push BOW in BOWS");
                bows.push_back(bow);

                ROS_INFO_STREAM("Immagine salvata con header " << image_msg->header.seq);
                toImgSeq.push_back(image_msg->header.seq);

                totalBOW++;
                ROS_INFO_STREAM("totalBOW = " << totalBOW);
               
                firstFrame_ = false;
            }
        }
        else
        {   
            ROS_WARN("--Image not descriptive enough, ignoring.");
        }

        if (totalBOW >= maxBOW_) 
        {
            // storage all descriptors and all image of the prerun
            ROS_INFO("Save BOWS in storage file");
            cv::FileStorage st;
	    // WRITE O APPEND??
            st.open(storagePath_,cv::FileStorage::WRITE);
            st << "Storage" << bows;
            st.release();

            ROS_INFO("Save Image seq in image file");
            cv::FileStorage im;
            im.open(imagePath_,cv::FileStorage::WRITE);
            im << "Image" << toImgSeq;
            im.release();


            shutdown();
        }
    }    // end class implemtation OpenFABMapPreRun
