#include "../include/openfabmap2_ros.h"
#include <iostream>


    /////////////////////////////////////
    //// *** OpenFABMap2 ROS BASE ***////
    /////////////////////////////////////
    //// Constructor
    // Pre:
    // Post: --Load parameters
    OpenFABMap2::OpenFABMap2(ros::NodeHandle nh) :
        nh_(nh),
        it_(nh),
        firstFrame_(true),
        visualise_(false),
        working_(true),
        saveQuit_(false)
    {
        // TODO: finish implementing parameter server
        // Read private parameters
        ros::NodeHandle local_nh_("~");
        local_nh_.param<std::string>("vocab", vocabPath_, "vocab.yml");
        local_nh_.param<std::string>("clTree", clTreePath_, "clTree.yml");
        local_nh_.param<std::string>("trainbows", trainbowsPath_, "trainbows.yml");
        local_nh_.param<std::string>("transport", transport_, "raw");
        local_nh_.param<bool>("visualise", visualise_, false);
        local_nh_.param<int>("MinDescriptorCount", minDescriptorCount_, 50);

        // Read node parameters
        imgTopic_ = nh_.resolveName("image");

        // Initialise feature method

        //////////
        // Surf parameters that may be used for both 'detector' and 'extractor'
        int surf_hessian_threshold, surf_num_octaves, surf_num_octave_layers, surf_upright, surf_extended;

        local_nh_.param<int>("HessianThreshold", surf_hessian_threshold, 1000);
        local_nh_.param<int>("NumOctaves", surf_num_octaves, 4);
        local_nh_.param<int>("NumOctaveLayers", surf_num_octave_layers, 2);
        local_nh_.param<int>("Extended", surf_extended, 0);
        local_nh_.param<int>("Upright", surf_upright, 1);

        //////////
        //create common feature detector
        std::string detectorType;
        local_nh_.param<std::string>("DetectorType", detectorType, "FAST");
        if (detectorType == "STAR") {
            int star_max_size, star_response, star_line_threshold, star_line_binarized, star_suppression;
            local_nh_.param<int>("MaxSize", star_max_size, 32);
            local_nh_.param<int>("Response", star_response, 10);
            local_nh_.param<int>("LineThreshold", star_line_threshold, 18);
            local_nh_.param<int>("LineBinarized", star_line_binarized, 18);
            local_nh_.param<int>("Suppression", star_suppression, 20);
            detector = new cv::StarFeatureDetector(
                star_max_size,
                star_response,
                star_line_threshold,
                star_line_binarized,
                star_suppression);

        } else if (detectorType == "FAST") {
            int fast_threshold, fast_non_max_suppression;
            local_nh_.param<int>("Threshold", fast_threshold, 50);
            local_nh_.param<int>("NonMaxSuppression", fast_non_max_suppression, 1);
            detector = new cv::FastFeatureDetector(
                fast_threshold,
                fast_non_max_suppression > 0);

        } else if (detectorType == "SURF") {
            detector = new cv::SURF(
                surf_hessian_threshold,
                surf_num_octaves,
                surf_num_octave_layers,
                surf_extended > 0,
                surf_upright > 0);

        } else if (detectorType == "SIFT") {
            int sift_nfeatures, sift_num_octave_layers;
            double sift_threshold, sift_edge_threshold, sift_sigma;
            local_nh_.param<int>("NumFeatures", sift_nfeatures, 0);
            local_nh_.param<int>("NumOctaveLayers", sift_num_octave_layers, 3);
            local_nh_.param<double>("Threshold", sift_threshold, 0.04);
            local_nh_.param<double>("EdgeThreshold", sift_edge_threshold, 10);
            local_nh_.param<double>("Sigma", sift_sigma, 1.6);
            detector = new cv::SIFT(
                sift_nfeatures,
                sift_num_octave_layers,
                sift_threshold,
                sift_edge_threshold,
                sift_sigma);

        } else {
            int mser_delta, mser_min_area, mser_max_area, mser_max_evolution, mser_edge_blur_size;
            double mser_max_variation, mser_min_diversity, mser_area_threshold, mser_min_margin;
            local_nh_.param<int>("Delta", mser_delta, 5);
            local_nh_.param<int>("MinArea", mser_min_area, 60);
            local_nh_.param<int>("MaxArea", mser_max_area, 14400);
            local_nh_.param<double>("MaxVariation", mser_max_variation, 0.25);
            local_nh_.param<double>("MinDiversity", mser_min_diversity, 0.2);
            local_nh_.param<int>("MaxEvolution", mser_max_evolution, 200);
            local_nh_.param<double>("AreaThreshold", mser_area_threshold, 1.01);
            local_nh_.param<double>("MinMargin", mser_min_margin, 0.003);
            local_nh_.param<int>("EdgeBlurSize", mser_edge_blur_size, 5);
            detector = new cv::MSER(
                mser_delta,
                mser_min_area,
                mser_max_area,
                mser_max_variation,
                mser_min_diversity,
                mser_max_evolution,
                mser_area_threshold,
                mser_min_margin,
                mser_edge_blur_size);
        }

        //create common descriptor extractor
        if (detectorType == "SIFT") {
            extractor = new cv::SIFT();
        } else {
            extractor = new cv::SURF(
                surf_hessian_threshold,
                surf_num_octaves,
                surf_num_octave_layers,
                surf_extended > 0,
                surf_upright > 0);
        }

        matcher = new cv::FlannBasedMatcher(); //can cv::BFMatcher be faster in our case?
        bide = new cv::BOWImgDescriptorExtractor(extractor, matcher);
    }

    // Destructor
    OpenFABMap2::~OpenFABMap2()
    {
    }

    //// Set Callback
    // Pre: --Valid 'imgTopic_' exists
    // Post: --Subscribes for Images with 'processImgCallback'
    void OpenFABMap2::subscribeToImages()
    {
        // Subscribe to images
        ROS_INFO("Subscribing to:\n\t* %s", imgTopic_.c_str());

        sub_ = it_.subscribe(imgTopic_, 1, &OpenFABMap2::processImgCallback, this,
            transport_);
    }

    //// Working Check
    // Pre: none
    // Post: none
    bool OpenFABMap2::isWorking() const
    {
        return working_;
    }
    // end class implemtation OpenFABMap2



