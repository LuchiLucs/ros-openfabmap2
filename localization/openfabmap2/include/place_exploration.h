#ifndef _place_exploration_ros_h
#define _place_exploration_ros_h

#include "openfabmap2_ros.h"

	/// PreRunning OpenFABMap2
	class FABMapPreRun : public OpenFABMap2
	{
	public:
		FABMapPreRun(ros::NodeHandle nh);
		~FABMapPreRun();
		void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg);
		bool loadCodebook();
		void shutdown();


	private:

		int maxMatches_;
		double minMatchValue_;
		bool disable_self_match_;
		int self_match_window_;
		bool disable_unknown_match_;
		bool only_new_places_;
		int maxBOW_;

		int totalBOW;
		std::vector<int> toImgSeq;

		std::string storagePath_;
		std::string imagePath_;
	};
#endif
