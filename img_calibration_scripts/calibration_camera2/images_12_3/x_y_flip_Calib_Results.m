% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 1550.622735862057880 ; 1557.902483102389169 ];

%-- Principal point:
cc = [ 1292.692659692555480 ; 976.489604547285012 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.017455517133265 ; -0.000000000000000 ; -0.009111056390290 ; 0.014063090737057 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 21.081934776876089 ; 21.560862655085934 ];

%-- Principal point uncertainty:
cc_error = [ 20.900020225924305 ; 22.081485298562455 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.009447220378909 ; 0.000000000000000 ; 0.004059732320618 ; 0.003795744447849 ; 0.000000000000000 ];

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
est_dist = [ 1 ; 0 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ -1.961905e+00 ; -2.346188e+00 ; 1.995972e-01 ];
Tc_1  = [ 4.832786e+02 ; -8.209732e+02 ; 2.123771e+03 ];
omc_error_1 = [ 1.026481e-02 ; 1.215942e-02 ; 2.468344e-02 ];
Tc_error_1  = [ 2.945683e+01 ; 3.023945e+01 ; 3.402061e+01 ];

%-- Image #2:
omc_2 = [ -1.879812e+00 ; -2.378950e+00 ; 2.137132e-01 ];
Tc_2  = [ 5.075884e+02 ; -3.655731e+02 ; 2.117198e+03 ];
omc_error_2 = [ 8.942787e-03 ; 1.361575e-02 ; 2.587039e-02 ];
Tc_error_2  = [ 2.887481e+01 ; 3.020855e+01 ; 3.196036e+01 ];

%-- Image #3:
omc_3 = [ -1.915764e+00 ; -2.464899e+00 ; 5.170400e-02 ];
Tc_3  = [ 4.603486e+02 ; 1.792400e+02 ; 2.049855e+03 ];
omc_error_3 = [ 5.490014e-03 ; 1.437232e-02 ; 2.520224e-02 ];
Tc_error_3  = [ 2.840674e+01 ; 2.940145e+01 ; 2.881369e+01 ];

%-- Image #4:
omc_4 = [ -1.876018e+00 ; -2.417660e+00 ; 1.349804e-01 ];
Tc_4  = [ 4.581025e+02 ; 3.224290e+02 ; 2.088640e+03 ];
omc_error_4 = [ 5.662353e-03 ; 1.501760e-02 ; 2.637061e-02 ];
Tc_error_4  = [ 2.917876e+01 ; 3.009963e+01 ; 2.915949e+01 ];

%-- Image #5:
omc_5 = [ NaN ; NaN ; NaN ];
Tc_5  = [ NaN ; NaN ; NaN ];
omc_error_5 = [ NaN ; NaN ; NaN ];
Tc_error_5  = [ NaN ; NaN ; NaN ];

%-- Image #6:
omc_6 = [ -2.235654e+00 ; -1.418711e+00 ; -5.877749e-01 ];
Tc_6  = [ -1.217932e+03 ; -5.053789e+01 ; 1.851267e+03 ];
omc_error_6 = [ 1.406212e-02 ; 8.894843e-03 ; 1.963581e-02 ];
Tc_error_6  = [ 2.606252e+01 ; 2.925285e+01 ; 3.384293e+01 ];

%-- Image #7:
omc_7 = [ 2.024800e+00 ; 2.053074e+00 ; 9.314083e-01 ];
Tc_7  = [ -9.236350e+02 ; -5.546638e+02 ; 1.806704e+03 ];
omc_error_7 = [ 1.298755e-02 ; 1.142807e-02 ; 2.249654e-02 ];
Tc_error_7  = [ 2.584365e+01 ; 2.849400e+01 ; 3.135377e+01 ];

%-- Image #8:
omc_8 = [ NaN ; NaN ; NaN ];
Tc_8  = [ NaN ; NaN ; NaN ];
omc_error_8 = [ NaN ; NaN ; NaN ];
Tc_error_8  = [ NaN ; NaN ; NaN ];

%-- Image #9:
omc_9 = [ 1.327811e+00 ; 2.114430e+00 ; -1.320850e-01 ];
Tc_9  = [ -3.871824e+02 ; -1.107899e+02 ; 2.228695e+03 ];
omc_error_9 = [ 8.382395e-03 ; 1.135923e-02 ; 1.919257e-02 ];
Tc_error_9  = [ 3.007623e+01 ; 3.198559e+01 ; 2.817323e+01 ];

%-- Image #10:
omc_10 = [ -1.661969e+00 ; -2.361457e+00 ; 8.972426e-01 ];
Tc_10  = [ 3.013491e+02 ; 3.092888e+02 ; 2.360417e+03 ];
omc_error_10 = [ 9.988892e-03 ; 1.555303e-02 ; 2.656589e-02 ];
Tc_error_10  = [ 3.207452e+01 ; 3.387500e+01 ; 2.952192e+01 ];

%-- Image #11:
omc_11 = [ -2.150940e+00 ; -1.905541e+00 ; -8.602594e-01 ];
Tc_11  = [ 2.492573e+02 ; -3.772221e+02 ; 1.819744e+03 ];
omc_error_11 = [ 6.251538e-03 ; 1.405657e-02 ; 2.097787e-02 ];
Tc_error_11  = [ 2.525558e+01 ; 2.619565e+01 ; 2.979812e+01 ];

%-- Image #12:
omc_12 = [ 1.494819e+00 ; 2.104848e+00 ; 3.703081e-01 ];
Tc_12  = [ 1.084196e+03 ; -8.540486e+02 ; 2.095978e+03 ];
omc_error_12 = [ 1.303030e-02 ; 1.226587e-02 ; 1.555731e-02 ];
Tc_error_12  = [ 3.052328e+01 ; 3.132844e+01 ; 3.850492e+01 ];

%-- Image #13:
omc_13 = [ NaN ; NaN ; NaN ];
Tc_13  = [ NaN ; NaN ; NaN ];
omc_error_13 = [ NaN ; NaN ; NaN ];
Tc_error_13  = [ NaN ; NaN ; NaN ];

%-- Image #14:
omc_14 = [ 2.004172e+00 ; 2.092435e+00 ; -1.109830e-01 ];
Tc_14  = [ -2.138453e+03 ; -1.000319e+03 ; 2.910473e+03 ];
omc_error_14 = [ 8.094476e-03 ; 2.093009e-02 ; 2.692132e-02 ];
Tc_error_14  = [ 4.375901e+01 ; 4.903609e+01 ; 5.565993e+01 ];

%-- Image #15:
omc_15 = [ 2.099249e+00 ; 2.017085e+00 ; 2.274102e-01 ];
Tc_15  = [ -1.009781e+03 ; 7.002665e+02 ; 2.806904e+03 ];
omc_error_15 = [ 1.992859e-02 ; 1.444444e-02 ; 3.046998e-02 ];
Tc_error_15  = [ 3.878381e+01 ; 4.164172e+01 ; 4.587211e+01 ];

