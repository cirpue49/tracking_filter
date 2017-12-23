//
// Created by kosuke on 12/23/17.
//

#ifndef HELP_FILTER_GROUND_TRUTH_PACKAGE_H
#define HELP_FILTER_GROUND_TRUTH_PACKAGE_H

#include "Eigen/Dense"

class GroundTruthPackage {
public:
    long long timestamp_;

    enum SensorType{
        LASER,
        RADAR
    } sensor_type_;

    Eigen::VectorXd gt_values_;

};


#endif //HELP_FILTER_GROUND_TRUTH_PACKAGE_H
