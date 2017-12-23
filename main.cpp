
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ukf.h"
#include "ground_truth_package.h"
#include "measurement_package.h"
#include "tools.h"
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void check_arguments(int argc, char* argv[]) {
    string usage_instructions = "Usage instructions: ";
    usage_instructions += argv[0];
    usage_instructions += " path/to/input.txt output.txt";

    bool has_valid_args = false;

    // make sure the user has provided input and output files
    if (argc == 1) {
        cerr << usage_instructions << endl;
    } else if (argc == 2) {
        cerr << "Please include an output file.\n" << usage_instructions << endl;
    } else if (argc == 3) {
        has_valid_args = true;
    } else if (argc > 3) {
        cerr << "Too many arguments.\n" << usage_instructions << endl;
    }

    if (!has_valid_args) {
        exit(EXIT_FAILURE);
    }
}

void check_files(ifstream& in_file, string& in_name,
                 ofstream& out_file, string& out_name) {
    if (!in_file.is_open()) {
        cerr << "Cannot open input file: " << in_name << endl;
        exit(EXIT_FAILURE);
    }

    if (!out_file.is_open()) {
        cerr << "Cannot open output file: " << out_name << endl;
        exit(EXIT_FAILURE);
    }
}

void pdaUpdate(vector<UKF>& targets, MeasurementPackage meas_package){
    vector<double> gaussL;
    vector<vector<double>>cvPDA, ctrvPDA, rmPDA;
    for (int targetInd = 0; targetInd < targets.size(); targetInd++){
        double detS1 = fabs(targets[targetInd].lS_cv_.determinant());
        double detS2 = fabs(targets[targetInd].lS_ctrv_.determinant());
        double detS3 = fabs(targets[targetInd].lS_rm_.determinant());

        MatrixXd inS1 = targets[targetInd].lS_cv_.inverse();
        MatrixXd inS2 = targets[targetInd].lS_ctrv_.inverse();
        MatrixXd inS3 = targets[targetInd].lS_rm_.inverse();

        VectorXd zPred1 = targets[targetInd].zPredCVl_;
        VectorXd zPred2 = targets[targetInd].zPredCTRVl_;
        VectorXd zPred3 = targets[targetInd].zPredRMl_;

        VectorXd z_a = meas_package.raw_measurements_;
        VectorXd z_b = meas_package.dummy_measurements_;

        double tempLa_1 = exp(-1*(((z_a-zPred1).transpose()*inS1*(z_a-zPred1))(0))/2)/sqrt(((2*M_PI)*(2*M_PI)*detS1));
        double tempLa_2 = exp(-1*(((z_a-zPred2).transpose()*inS2*(z_a-zPred2))(0))/2)/sqrt(((2*M_PI)*(2*M_PI)*detS2));
        double tempLa_3 = exp(-1*(((z_a-zPred3).transpose()*inS1*(z_a-zPred3))(0))/2)/sqrt(((2*M_PI)*(2*M_PI)*detS3));

        double tempLb_1 = exp(-1*(((z_b-zPred1).transpose()*inS1*(z_b-zPred1))(0))/2)/sqrt(((2*M_PI)*(2*M_PI)*detS1));
        double tempLb_2 = exp(-1*(((z_b-zPred2).transpose()*inS2*(z_b-zPred2))(0))/2)/sqrt(((2*M_PI)*(2*M_PI)*detS2));
        double tempLb_3 = exp(-1*(((z_b-zPred3).transpose()*inS1*(z_b-zPred3))(0))/2)/sqrt(((2*M_PI)*(2*M_PI)*detS3));

        vector<double>tempCV   = {tempLa_1, tempLb_1};
        vector<double>tempCTRV = {tempLa_2, tempLb_2};
        vector<double>tempRM   = {tempLa_3, tempLb_3};

        cvPDA.push_back(tempCV);
        ctrvPDA.push_back(tempCTRV);
        rmPDA.push_back(tempRM);
//        cvPDA[targetInd].push_back(tempLa_1);
//        cvPDA[targetInd].push_back(tempLb_1);
//
//        ctrvPDA[targetInd].push_back(tempLa_2);
//        ctrvPDA[targetInd].push_back(tempLb_2);
//
//        rmPDA[targetInd].push_back(tempLa_3);
//        rmPDA[targetInd].push_back(tempLb_3);
    }



    // calculate joint association probability
    // TODO recursive or dynamic programinng, calculating combination
    double pD = 0.9;                                 // target, measurement
    double gammaCV1 = pD*pD*cvPDA[0][0]*cvPDA[1][0]; // 11 21
    double gammaCV2 = pD*pD*cvPDA[0][0]*cvPDA[1][1]; // 11 22
    double gammaCV3 = pD*pD*cvPDA[0][1]*cvPDA[1][0]; // 12 21
    double gammaCV4 = pD*pD*cvPDA[0][1]*cvPDA[1][1]; // 12 22

    double gammaCTRV1 = pD*pD*ctrvPDA[0][0]*ctrvPDA[1][0]; // 11 21
    double gammaCTRV2 = pD*pD*ctrvPDA[0][0]*ctrvPDA[1][1]; // 11 22
    double gammaCTRV3 = pD*pD*ctrvPDA[0][1]*ctrvPDA[1][0]; // 12 21
    double gammaCTRV4 = pD*pD*ctrvPDA[0][1]*ctrvPDA[1][1]; // 12 22

    double gammaRM1 = pD*pD*rmPDA[0][0]*rmPDA[1][0]; // 11 21
    double gammaRM2 = pD*pD*rmPDA[0][0]*rmPDA[1][1]; // 11 22
    double gammaRM3 = pD*pD*rmPDA[0][1]*rmPDA[1][0]; // 12 21
    double gammaRM4 = pD*pD*rmPDA[0][1]*rmPDA[1][1]; // 12 22

    double sumGammaCV   = gammaCV1   +gammaCV2   +gammaCV3   +gammaCV4;
    double sumGammaCTRV = gammaCTRV1 +gammaCTRV2 +gammaCTRV3 +gammaCTRV4;
    double sumGammaRM   = gammaRM1   +gammaRM2   +gammaRM3   +gammaRM4;

    double pCV11 = (gammaCV1+gammaCV2)/sumGammaCV;
    double pCV12 = (gammaCV3+gammaCV4)/sumGammaCV;
    double pCV21 = (gammaCV1+gammaCV3)/sumGammaCV;
    double pCV22 = (gammaCV2+gammaCV4)/sumGammaCV;

    double pCTRV11 = (gammaCTRV1+gammaCTRV2)/sumGammaCTRV;
    double pCTRV12 = (gammaCTRV3+gammaCTRV4)/sumGammaCTRV;
    double pCTRV21 = (gammaCTRV1+gammaCTRV3)/sumGammaCTRV;
    double pCTRV22 = (gammaCTRV2+gammaCTRV4)/sumGammaCTRV;

    double pRM11 = (gammaRM1+gammaRM2)/sumGammaRM;
    double pRM12 = (gammaRM3+gammaRM4)/sumGammaRM;
    double pRM21 = (gammaRM1+gammaRM3)/sumGammaRM;
    double pRM22 = (gammaRM2+gammaRM4)/sumGammaRM;


    UKF target1 = targets[0];
    UKF target2 = targets[1];
    VectorXd z_pred_cv1, z_pred_ctrv1, z_pred_rm1, z_pred_cv2, z_pred_ctrv2, z_pred_rm2;
    MatrixXd Scv1, Sctrv1, Srm1, Pcv1, Pctrv1, Prm1, Kcv1, Kctrv1, Krm1, Xcv1, Xctrv1, Xrm1;
    MatrixXd Scv2, Sctrv2, Srm2, Pcv2, Pctrv2, Prm2, Kcv2, Kctrv2, Krm2, Xcv2, Xctrv2, Xrm2;

    VectorXd z_raw = meas_package.raw_measurements_;
    VectorXd z_dum = meas_package.dummy_measurements_;

    z_pred_cv1 = target1.zPredCVl_;
    Scv1       = target1.lS_cv_;
    Xcv1       = target1.x_cv_;
    Pcv1       = target1.P_cv_;
    Kcv1      = target1.K_cv_;

    z_pred_cv2 = target2.zPredCVl_;
    Scv2       = target2.lS_cv_;
    Xcv2       = target2.x_cv_;
    Pcv2       = target2.P_cv_;
    Kcv2       = target2.K_cv_;

    z_pred_ctrv1 = target1.zPredCTRVl_;
    Sctrv1       = target1.lS_ctrv_;
    Xctrv1       = target1.x_ctrv_;
    Pctrv1       = target1.P_ctrv_;
    Kctrv1       = target1.K_ctrv_;

    z_pred_ctrv2 = target2.zPredCTRVl_;
    Sctrv2       = target2.lS_ctrv_;
    Xctrv2       = target2.x_ctrv_;
    Pctrv2       = target2.P_ctrv_;
    Kctrv2       = target2.K_ctrv_;

    z_pred_rm1 = target1.zPredRMl_;
    Srm1       = target1.lS_rm_;
    Xrm1       = target1.x_rm_;
    Prm1       = target1.P_rm_;
    Krm1       = target1.K_rm_;

    z_pred_rm2 = target1.zPredRMl_;
    Srm2       = target1.lS_rm_;
    Xrm2       = target1.x_rm_;
    Prm2       = target1.P_rm_;
    Krm2       = target1.K_rm_;


    //residual,
    VectorXd z_diff_cv1    = pCV11  *(z_raw - z_pred_cv1)   +pCV12  *(z_dum - z_pred_cv1);
    VectorXd z_diff_cv2    = pCV21  *(z_raw - z_pred_cv2)   +pCV22  *(z_dum - z_pred_cv2);
    VectorXd z_diff_ctrv1  = pCTRV11*(z_raw - z_pred_ctrv1) +pCTRV12*(z_dum - z_pred_ctrv1);
    VectorXd z_diff_ctrv2  = pCTRV21*(z_raw - z_pred_ctrv2) +pCTRV22*(z_dum - z_pred_ctrv2);
    VectorXd z_diff_rm1    = pRM11  *(z_raw - z_pred_rm1)   +pRM12  *(z_dum - z_pred_rm1);
    VectorXd z_diff_rm2    = pRM21  *(z_raw - z_pred_rm2)   +pRM22  *(z_dum - z_pred_rm2);

    //update state mean and covariance matrix
    Xcv1   = Xcv1   + Kcv1   * z_diff_cv1;
    Xcv2   = Xcv2   + Kcv2   * z_diff_cv2;
    Xctrv1 = Xctrv1 + Kctrv1 * z_diff_ctrv1;
    Xctrv2 = Xctrv2 + Kctrv2 * z_diff_ctrv2;
    Xrm1   = Xrm1   + Krm1   * z_diff_rm1;
    Xrm2   = Xrm2   + Krm2   * z_diff_rm2;

    while (Xcv1(3)> M_PI)   Xcv1(3) -= 2.*M_PI;
    while (Xcv1(3)<-M_PI)   Xcv1(3) += 2.*M_PI;
    while (Xcv2(3)> M_PI)   Xcv2(3) -= 2.*M_PI;
    while (Xcv2(3)<-M_PI)   Xcv2(3) += 2.*M_PI;
    while (Xctrv1(3)> M_PI) Xctrv1(3) -= 2.*M_PI;
    while (Xctrv1(3)<-M_PI) Xctrv1(3) += 2.*M_PI;
    while (Xctrv2(3)> M_PI) Xctrv2(3) -= 2.*M_PI;
    while (Xctrv2(3)<-M_PI) Xctrv2(3) += 2.*M_PI;
    while (Xrm1(3)> M_PI)   Xrm1(3) -= 2.*M_PI;
    while (Xrm1(3)<-M_PI)   Xrm1(3) += 2.*M_PI;
    while (Xrm2(3)> M_PI)   Xrm2(3) -= 2.*M_PI;
    while (Xrm2(3)<-M_PI)   Xrm2(3) += 2.*M_PI;


    // TODO use beta0?
    MatrixXd diffCV1   = (pCV11    *(z_raw - z_pred_cv1)   * z_diff_cv1.transpose()) +
                         (pCV12    *(z_dum - z_pred_cv1)   * z_diff_cv1.transpose());
    MatrixXd diffCV2   = (pCV21    *(z_raw - z_pred_cv2)   * z_diff_cv2.transpose()) +
                         (pCV22    *(z_dum - z_pred_cv2)   * z_diff_cv2.transpose());
    MatrixXd diffCTRV1 = (pCTRV11  *(z_raw - z_pred_ctrv1)  * z_diff_ctrv1.transpose()) +
                         (pCTRV12  *(z_dum - z_pred_ctrv1) * z_diff_ctrv1.transpose());
    MatrixXd diffCTRV2 = (pCTRV21  *(z_raw - z_pred_ctrv2) * z_diff_ctrv2.transpose()) +
                         (pCTRV22  *(z_dum - z_pred_ctrv2) * z_diff_ctrv2.transpose());
    MatrixXd diffRM1   = (pRM11    *(z_raw - z_pred_rm1)   * z_diff_rm1.transpose())   +
                         (pRM12    *(z_dum - z_pred_rm1)   * z_diff_rm1.transpose());
    MatrixXd diffRM2   = (pRM21    *(z_raw - z_pred_rm2)   * z_diff_rm2.transpose())   +
                         (pRM22    *(z_dum - z_pred_rm2)   * z_diff_rm2.transpose());

    Pcv1   = Pcv1   - Kcv1*Scv1*Kcv1.transpose()       + Kcv1*diffCV1*Kcv1.transpose();
    Pcv2   = Pcv2   - Kcv2*Scv2*Kcv2.transpose()       + Kcv2*diffCV2*Kcv2.transpose();
    Pctrv1 = Pctrv1 - Kctrv1*Sctrv1*Kctrv1.transpose() + Kctrv1*diffCTRV1*Kctrv1.transpose();
    Pctrv2 = Pctrv2 - Kctrv2*Sctrv2*Kctrv2.transpose() + Kctrv2*diffCTRV2*Kctrv2.transpose();;
    Prm1   = Prm1   - Krm1*Srm1*Krm1.transpose()       + Krm1*diffRM1*Krm1.transpose();
    Prm2   = Prm2   - Krm2*Srm2*Krm2.transpose()       + Krm2*diffRM2*Krm2.transpose();


    /*****************************************************************************
    *  Update model parameters
    ****************************************************************************/
    target1.x_cv_.col(0)  = Xcv1;
    target2.x_cv_.col(0)  = Xcv2;
    target1.P_cv_         = Pcv1;
    target2.P_cv_         = Pcv2;

    target1.x_ctrv_.col(0)  = Xctrv1;
    target2.x_ctrv_.col(0)  = Xctrv2;
    target1.P_ctrv_         = Pctrv1;
    target2.P_ctrv_         = Pctrv2;

    target1.x_rm_.col(0)    = Xrm1;
    target2.x_rm_.col(0)    = Xrm2;
    target1.P_rm_           = Prm1;
    target2.P_rm_           = Prm2;


}

int main(int argc, char* argv[]) {

    cout << 123 << endl;
    check_arguments(argc, argv);

    string in_file_name_ = argv[1];
    ifstream in_file_(in_file_name_.c_str(), ifstream::in);

    string out_file_name_ = argv[2];
    ofstream out_file_(out_file_name_.c_str(), ofstream::out);

    check_files(in_file_, in_file_name_, out_file_, out_file_name_);

    /**********************************************
     *  Set Measurements                          *
     **********************************************/

    vector<MeasurementPackage> measurement_pack_list;
    vector<MeasurementPackage> dummy_measurement_pack_list;
    vector<GroundTruthPackage> gt_pack_list;

    string line;

    // prep the measurement packages (each line represents a measurement at a
    // timestamp)
    while (getline(in_file_, line)) {
        string sensor_type;
        MeasurementPackage meas_package;
        GroundTruthPackage gt_package;
        istringstream iss(line);
        long long timestamp;

        // reads first element from the current line
        iss >> sensor_type;

        if (sensor_type.compare("L") == 0) {
            // laser measurement

            // read measurements at this timestamp
            meas_package.sensor_type_ = MeasurementPackage::LASER;
            meas_package.raw_measurements_   = VectorXd(2);
            meas_package.dummy_measurements_ = VectorXd(2);
            float px;
            float py;
            iss >> px;
            iss >> py;
            meas_package.raw_measurements_  << px, py;
            meas_package.dummy_measurements_ << px*cos(M_PI/100) - py*sin(M_PI/100) , px*sin(M_PI/100) + py*cos(M_PI/100) + 1;
            iss >> timestamp;
            meas_package.timestamp_  = timestamp;
            measurement_pack_list.push_back(meas_package);
        } else if (sensor_type.compare("R") == 0) {
            continue;
        }

        // read ground truth data to compare later
        float x_gt;
        float y_gt;
        float vx_gt;
        float vy_gt;
        iss >> x_gt;
        iss >> y_gt;
        iss >> vx_gt;
        iss >> vy_gt;
        gt_package.gt_values_ = VectorXd(4);
        gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
        gt_pack_list.push_back(gt_package);
    }


    // Create a UKF instance
    UKF ukf1;
    // assuming there are always 2 targets and 2 measurements
    UKF ukf2;

    // used to compute the RMSE later
    vector<VectorXd> estimations;
    vector<VectorXd> ground_truth;

    // start filtering from the second frame (the speed is unknown in the first
    // frame)

    size_t number_of_measurements = measurement_pack_list.size();

    // column names for output file
//    out_file_ << "time_stamp" << "\t";
//    out_file_ << "px" << "\t";
//    out_file_ << "py" << "\t";
//    out_file_ << "v" << "\t";
//    out_file_ << "yaw_angle" << "\t";
//    out_file_ << "yaw_rate" << "\t";
//    out_file_ << "sensor_type" << "\t";
//    out_file_ << "NIS" << "\t";
//    out_file_ << "px_measured" << "\t";
//    out_file_ << "py_measured" << "\t";
//    out_file_ << "px_true" << "\t";
//    out_file_ << "py_true" << "\t";
//    out_file_ << "vx_true" << "\t";
//    out_file_ << "vy_true" << "\n";


    for (size_t k = 0; k < number_of_measurements; ++k) {
        // Call the UKF-based fusion
        VectorXd z1 = measurement_pack_list[k].raw_measurements_;
        VectorXd z2 = measurement_pack_list[k].dummy_measurements_;
        if(!ukf1.is_initialized_){
            ukf1.Initialize(measurement_pack_list[k], z1);
            ukf2.Initialize(measurement_pack_list[k], z2);
            ukf1.is_initialized_ = true;
            ukf2.is_initialized_ = true;
            cout << "ukf1"<<endl << ukf1.x_merge_ << endl;
            cout << "ukf2"<<endl << ukf2.x_merge_ << endl;

            continue;
        }

        ukf1.ProcessMeasurement(measurement_pack_list[k], z1);
        ukf2.ProcessMeasurement(measurement_pack_list[k], z2);

        vector<UKF> targets;
        targets.push_back(ukf1);
        targets.push_back(ukf2);
        pdaUpdate(targets, measurement_pack_list[k]);

        ukf1.PostProcessMeasurement(z1);
        ukf2.PostProcessMeasurement(z2);

        cout << "ukf1"<<endl << ukf1.x_merge_ << endl;
        cout << "ukf2"<<endl << ukf2.x_merge_ << endl;


        out_file_ << 1 << "\t";
        // output the estimation
        out_file_ << ukf1.x_merge_(0, 0) << "\t"; // pos1 - est
        out_file_ << ukf1.x_merge_(1, 0) << "\t"; // pos2 - est
        out_file_ << ukf1.x_merge_(2, 0) << "\t"; // vel_abs -est
        out_file_ << ukf1.x_merge_(3, 0) << "\t"; // yaw_angle -est
        out_file_ << ukf1.x_merge_(4, 0) << "\t"; // yaw_rate -est

//        out_file_ << ukf.x_rm_(0, 0) << "\t"; // pos1 - est
//        out_file_ << ukf.x_rm_(1, 0) << "\t"; // pos2 - est
//        out_file_ << ukf.x_rm_(2, 0) << "\t"; // vel_abs -est
//        out_file_ << ukf.x_rm_(3, 0) << "\t"; // yaw_angle -est
//        out_file_ << ukf.x_rm_(4, 0) << "\t"; // yaw_rate -est

        // output the measurements only lidar
        out_file_ << "lidar" << "\t";
        out_file_ << 1 << "\t";

        double x = measurement_pack_list[k].raw_measurements_(0);
        double y = measurement_pack_list[k].raw_measurements_(1);

        // p1 - meas
        out_file_ << x << "\t";

        // p2 - meas
        out_file_ << y << "\t";

        // dummy input
        out_file_ << x*cos(M_PI/100) - y*sin(M_PI/100)  << "\t";
        out_file_ << x*sin(M_PI/100) + y*cos(M_PI/100) + 1 << "\t";


        // output the ground truth packages
        out_file_ << gt_pack_list[k].gt_values_(0) << "\t";
        out_file_ << gt_pack_list[k].gt_values_(1) << "\t";
        out_file_ << gt_pack_list[k].gt_values_(2) << "\t";
        out_file_ << gt_pack_list[k].gt_values_(3) << "\t";

        out_file_ << ukf2.x_merge_(0, 0) << "\t"; // pos1 - est
        out_file_ << ukf2.x_merge_(1, 0) << "\t"; // pos2 - est
        out_file_ << ukf2.x_merge_(2, 0) << "\t"; // vel_abs -est
        out_file_ << ukf2.x_merge_(3, 0) << "\t"; // yaw_angle -est
        out_file_ << ukf2.x_merge_(4, 0) << "\n"; // yaw_rate -est

        // output the NIS values

//        if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::LASER) {
//            out_file_ << ukf.NIS_laser_ << "\n";
//        } else if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::RADAR) {
//            out_file_ << ukf.NIS_radar_ << "\n";
//        }


        // convert ukf x vector to cartesian to compare to ground truth
        VectorXd ukf_x_cartesian_ = VectorXd(4);

        float x_estimate_ = ukf1.x_merge_(0, 0);
        float y_estimate_ = ukf1.x_merge_(1, 0);
        float vx_estimate_ = ukf1.x_merge_(2, 0) * cos(ukf1.x_merge_(3, 0));
        float vy_estimate_ = ukf1.x_merge_(2, 0) * sin(ukf1.x_merge_(3, 0));

        ukf_x_cartesian_ << x_estimate_, y_estimate_, vx_estimate_, vy_estimate_;

        estimations.push_back(ukf_x_cartesian_);
        ground_truth.push_back(gt_pack_list[k].gt_values_);

    }

    // compute the accuracy (RMSE)
    Tools tools;
    cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;
//    cout << "Empty count: "<<ukf1.count_empty_<<endl;

    // close files
    if (out_file_.is_open()) {
        out_file_.close();
    }

    if (in_file_.is_open()) {
        in_file_.close();
    }

    cout << "Done!" << endl;
    return 0;
}

//
//Accuracy - RMSE:
//0.0886119
//0.100177
//0.570503
//0.259474#include <iostream>

