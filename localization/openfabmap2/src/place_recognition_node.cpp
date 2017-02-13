#include <ros/ros.h>
#include "../include/place_recognition.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "place_recognition_node");

    // Running
    ros::NodeHandle priv_nh("~");
    double sampleRate_;
    priv_nh.param<double>("sampleRate", sampleRate_, 100);
    ros::Rate r(sampleRate_);

    ros::NodeHandle nh_run;
    FABMapRun oFABMap2_run(nh_run);

    ROS_INFO_STREAM("Node sampling rate set to: " << sampleRate_ << "Hz");

    ROS_INFO("------------- FABMapRun started ---------------");

    while (nh_run.ok() && oFABMap2_run.isWorking())
    {
        ros::spinOnce();
        r.sleep();
    }

    return 0;
}
