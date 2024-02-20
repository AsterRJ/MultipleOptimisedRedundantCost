#include <ros/ros.h>
#include "std_msgs/String.h"
#include "kmriiwa_nullspace/nullspace_calcs.h"
#include <sensor_msgs/JointState.h>
#include <cstdlib>
#include <sstream>
#include <string>
#include <std_msgs/Float64MultiArray.h>


// Author: Alastair Poole
// Email: alastair.poole@strath.ac.uk
// Stragclyde/SEARCH lab

// reads LBR-iiwa joint data and streams a length 10 optimisation vector that considers;
// 1. Singularity proximity,
// 2. Condition number,
// 3. Proximity to joint limits.

// TODO: read params from param server & create auto-diff functions.

ros::Publisher redundancy_vec_pub, metric_pub;
NullSpaceCalc ns;


void joint_space_cb(const sensor_msgs::JointState::ConstPtr & msg){
  Eigen::Matrix<double,7,1> q, desc_vec;
  Eigen::Matrix<double,3,1> c_vec;
  c_vec.setZero();
  q<<msg->position[0], msg->position[1],msg->position[2],msg->position[3],msg->position[4],msg->position[5],msg->position[6];

  ns.calculate_descent(q,c_vec, desc_vec);
  std_msgs::Float64MultiArray c_msg;
  for (int i = 0; i<3; i++){c_msg.data.push_back(c_vec(i));}
  metric_pub.publish(c_msg);

  std_msgs::Float64MultiArray desc_msg;
  for (int i = 0; i<3; i++){desc_msg.data.push_back(desc_vec(i));}
  desc_msg.data.push_back(0); desc_msg.data.push_back(0); desc_msg.data.push_back(0);
  // accom for last three axes - the parser I used couldnt handle static links, however the base's projections should be handled on the cartesian -> joint server side
  redundancy_vec_pub.publish(desc_msg);
}



int main(int argc, char **argv)
{

  ros::init(argc, argv, "joint_space_optimiser");
  ros::NodeHandle n;
  redundancy_vec_pub = n.advertise<std_msgs::Float64MultiArray>("secondary_velocity", 1);
  metric_pub = n.advertise<std_msgs::Float64MultiArray>("joint_cost_fns", 10);
  ros::Subscriber joint_sub = n.subscribe("kmriiwa/arm/joint_states", 1, joint_space_cb);
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }


  return 0;
}

