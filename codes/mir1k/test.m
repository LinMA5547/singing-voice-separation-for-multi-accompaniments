% %% unit test 2 - best setting:
% % context window size
% context_win = 3;
% % hidden units
% hidden_units = 1000;
% num_layers = 3;
% isdropout = 0;
% % RNN temporal connection
% isRNN = 2;
% % One output source or two
% iscleanonly = 0;
% % Circular shift step
% circular_step = 10000;
% % normalize input as L1 norm = 1
% isinputL1 = 0;
% % 0: MFCC, 1: logmel, 2: spectra
% MFCCorlogMelorSpectrum = 2;
% % feature frame rate
% framerate = 64;
% % discriminative training gamma parameter
% pos_neg_r = 0.05;
% % Last layer - linear or nonlinear
% outputnonlinear = 0;
% % soft mask obj
% softabs = 1;
% % 0: logistic, 1: tanh, 2: RELU
% act = 2;
% % constant for avoiding numerical problems
% const = 1e-10;
% % constant for avoiding numerical problems
% const2 = 0.001;
% % 0: not using GPU, 1: using GPU
% isGPU = 0;
% 
% train_mode = 0;
% % 0:'softlinear',1:'softabs', 2:'softquad', 3:'softabs_const',
% % 4:'softabs_kl_const'
% opt = 1;
% 
% train_mir1k_demo(context_win, hidden_units, num_layers, isdropout, ...
%     isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
%     framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
%     const2, isGPU)
savedir='results';
  fprintf('DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_DRNN_\n');
iter=1;
stage = 'testall';
modelname='model_test';
load([savedir,modelname, filesep, 'model_',num2str(iter),'.mat']);
test_mir1k_general_kl_bss3(modelname, theta, eI, stage, iter);

% iter=2;
%  fprintf('relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_relu_\n');
% 
% load([savedir,modelname, filesep, 'model_',num2str(iter),'.mat']);
% test_mir1k_general_kl_bss3(modelname, theta, eI, stage, iter);
%   fprintf('LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_LOG_\n');

% iter=2;
% 
% load([savedir,modelname, filesep, 'model_',num2str(iter),'.mat']);
% test_mir1k_general_kl_bss3(modelname, theta, eI, stage, iter);