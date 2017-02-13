#ifndef _learn_ros_h
#define _learn_ros_h

#include "openfabmap2_ros.h"

	//// Learning OpenFABMap2
	class FABMapLearn : public OpenFABMap2
	{
	public:
		FABMapLearn(ros::NodeHandle nh);
		~FABMapLearn();

		void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg);
		void findWords();
		void saveCodebook();
		void shutdown();

	private:
		int trainCount_;
		int maxImages_;
		double clusterSize_;
		double lowerInformationBound_;

		std::vector<cv_bridge::CvImagePtr> framesSampled;

		cv::Mat descriptors;
		cv::Mat bows;
		cv::of2::BOWMSCTrainer trainer;
		cv::of2::ChowLiuTree tree;
	};

#endif
