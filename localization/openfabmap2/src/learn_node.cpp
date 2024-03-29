#include <ros/ros.h>
#include "../include/learn.h"

int main(int argc, char **argv)
{
	ros::init(argc, argv, "learn_node");

	// Learning
	ros::NodeHandle priv_nh("~");
	double sampleRate_;
	priv_nh.param<double>("sampleRate", sampleRate_, 100);
	ros::Rate r(sampleRate_);

	ros::NodeHandle nh_learn;
	FABMapLearn oFABMap2_learn(nh_learn);

        std::cout << "------------- FABMapLearn started ---------------" << std::endl;

	ROS_INFO_STREAM("Node sampling rate set to: " << sampleRate_ << "Hz");
	while (nh_learn.ok() && oFABMap2_learn.isWorking())
	{
		ros::spinOnce();
		r.sleep();
	}

	ROS_INFO("Node closed.........");

	return 0;
}
