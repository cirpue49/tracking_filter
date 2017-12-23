//
// Created by kosuke on 12/23/17.
//

#ifndef HELP_FILTER_TOOLS_H
#define HELP_FILTER_TOOLS_H

#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
    /**
    * Constructor.
    */
    Tools();

    /**
    * Destructor.
    */
    virtual ~Tools();

    /**
    * A helper method to calculate RMSE.
    */
    VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

};



#endif //HELP_FILTER_TOOLS_H
