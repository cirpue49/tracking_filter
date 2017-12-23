//
// Created by kosuke on 12/23/17.
//

#ifndef HELP_FILTER_MEASUREMENT_PACKAGE_H
#define HELP_FILTER_MEASUREMENT_PACKAGE_H

#include "Eigen/Dense"

class MeasurementPackage {
public:
    long timestamp_;

    enum SensorType{
        LASER,
        RADAR
    } sensor_type_;

    Eigen::VectorXd raw_measurements_;
    Eigen::VectorXd dummy_measurements_;



};


#endif //HELP_FILTER_MEASUREMENT_PACKAGE_H
