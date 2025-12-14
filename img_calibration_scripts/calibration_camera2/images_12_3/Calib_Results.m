% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 1582.397776055927579 ; 1575.743790914296596 ];

%-- Principal point:
cc = [ 1234.305128192135726 ; 1030.290677523951899 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.073381208418122 ; -0.135671774884149 ; -0.002101009138053 ; -0.001467727198030 ; 0.101247113514878 ];

%-- Focal length uncertainty:
fc_error = [ 26.040428182963321 ; 26.469526193489276 ];

%-- Principal point uncertainty:
cc_error = [ 25.786388001462459 ; 25.061840782054013 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.031478908651698 ; 0.092116705440198 ; 0.004599893216985 ; 0.004728019875482 ; 0.101209364073095 ];

%-- Image size:
nx = 2448;
ny = 2048;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 15;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 1 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ 3.094991e+00 ; 2.826741e-01 ; -2.496401e-01 ];
Tc_1  = [ 4.228085e+02 ; -1.454464e+02 ; 2.126629e+03 ];
omc_error_1 = [ 1.917552e-02 ; 4.584713e-03 ; 2.952471e-02 ];
Tc_error_1  = [ 3.528537e+01 ; 3.424524e+01 ; 3.653962e+01 ];

%-- Image #2:
omc_2 = [ 3.093916e+00 ; 3.641325e-01 ; -2.999336e-01 ];
Tc_2  = [ 4.082526e+02 ; 3.028855e+02 ; 2.142745e+03 ];
omc_error_2 = [ 1.820652e-02 ; 3.492235e-03 ; 3.173582e-02 ];
Tc_error_2  = [ 3.526639e+01 ; 3.422913e+01 ; 3.776335e+01 ];

%-- Image #3:
omc_3 = [ 3.115937e+00 ; 3.735034e-01 ; -1.372766e-01 ];
Tc_3  = [ 3.577204e+02 ; 8.533311e+02 ; 2.109318e+03 ];
omc_error_3 = [ 1.882031e-02 ; 4.664482e-03 ; 2.905804e-02 ];
Tc_error_3  = [ 3.608543e+01 ; 3.397869e+01 ; 4.092949e+01 ];

%-- Image #4:
omc_4 = [ NaN ; NaN ; NaN ];
Tc_4  = [ NaN ; NaN ; NaN ];
omc_error_4 = [ NaN ; NaN ; NaN ];
Tc_error_4  = [ NaN ; NaN ; NaN ];

%-- Image #5:
omc_5 = [ NaN ; NaN ; NaN ];
Tc_5  = [ NaN ; NaN ; NaN ];
omc_error_5 = [ NaN ; NaN ; NaN ];
Tc_error_5  = [ NaN ; NaN ; NaN ];

%-- Image #6:
omc_6 = [ -2.447635e+00 ; 5.656462e-01 ; 6.688532e-02 ];
Tc_6  = [ -8.391091e+02 ; 4.309382e+02 ; 2.401024e+03 ];
omc_error_6 = [ 1.466604e-02 ; 1.025628e-02 ; 2.259506e-02 ];
Tc_error_6  = [ 3.995807e+01 ; 3.887123e+01 ; 4.397412e+01 ];

%-- Image #7:
omc_7 = [ -2.700896e+00 ; 2.223513e-02 ; -7.475717e-01 ];
Tc_7  = [ -9.096090e+02 ; 9.163231e+01 ; 2.133729e+03 ];
omc_error_7 = [ 1.897736e-02 ; 5.206129e-03 ; 2.353282e-02 ];
Tc_error_7  = [ 3.613631e+01 ; 3.521520e+01 ; 4.232053e+01 ];

%-- Image #8:
omc_8 = [ NaN ; NaN ; NaN ];
Tc_8  = [ NaN ; NaN ; NaN ];
omc_error_8 = [ NaN ; NaN ; NaN ];
Tc_error_8  = [ NaN ; NaN ; NaN ];

%-- Image #9:
omc_9 = [ 2.569202e+00 ; 5.814432e-01 ; 4.930714e-01 ];
Tc_9  = [ -5.394007e+02 ; 4.259220e+02 ; 1.876048e+03 ];
omc_error_9 = [ 1.667186e-02 ; 5.342946e-03 ; 2.311969e-02 ];
Tc_error_9  = [ 3.137228e+01 ; 3.053236e+01 ; 3.841261e+01 ];

%-- Image #10:
omc_10 = [ 2.634100e+00 ; 4.776887e-01 ; -7.138100e-01 ];
Tc_10  = [ 7.724086e+01 ; 9.086253e+02 ; 2.216064e+03 ];
omc_error_10 = [ 1.818146e-02 ; 6.372496e-03 ; 2.659150e-02 ];
Tc_error_10  = [ 3.781804e+01 ; 3.607014e+01 ; 4.084709e+01 ];

%-- Image #11:
omc_11 = [ NaN ; NaN ; NaN ];
Tc_11  = [ NaN ; NaN ; NaN ];
omc_error_11 = [ NaN ; NaN ; NaN ];
Tc_error_11  = [ NaN ; NaN ; NaN ];

%-- Image #12:
omc_12 = [ NaN ; NaN ; NaN ];
Tc_12  = [ NaN ; NaN ; NaN ];
omc_error_12 = [ NaN ; NaN ; NaN ];
Tc_error_12  = [ NaN ; NaN ; NaN ];

%-- Image #13:
omc_13 = [ NaN ; NaN ; NaN ];
Tc_13  = [ NaN ; NaN ; NaN ];
omc_error_13 = [ NaN ; NaN ; NaN ];
Tc_error_13  = [ NaN ; NaN ; NaN ];

%-- Image #14:
omc_14 = [ 2.927013e+00 ; 7.692044e-02 ; 1.019312e-01 ];
Tc_14  = [ -2.057008e+03 ; -3.762784e+02 ; 2.914400e+03 ];
omc_error_14 = [ 1.840071e-02 ; 1.081238e-02 ; 3.842762e-02 ];
Tc_error_14  = [ 5.081961e+01 ; 5.243031e+01 ; 7.323334e+01 ];

%-- Image #15:
omc_15 = [ 2.932885e+00 ; -1.009586e-01 ; 3.014059e-01 ];
Tc_15  = [ -8.545470e+02 ; 1.362463e+03 ; 2.893144e+03 ];
omc_error_15 = [ 3.090432e-02 ; 8.342645e-03 ; 4.908181e-02 ];
Tc_error_15  = [ 4.983136e+01 ; 4.753363e+01 ; 6.647947e+01 ];

