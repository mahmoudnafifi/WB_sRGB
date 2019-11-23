%% Evaluation examples
%
% Copyright (c) 2018-present, Mahmoud Afifi
% York University, Canada
% mafifi@eecs.yorku.ca | m.3afifi@gmail.com
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
% All rights reserved.
%
% Please cite the following work if this program is used:
% Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown, 
% "When color constancy goes wrong: Correcting improperly white-balanced 
% images", CVPR 2019.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
addpath('evaluation')

%% Please remove this part when you use your method
load(fullfile('models','WB_model.mat')); 
model.gamut_mapping = 2;


%% Example1 (RenderedWB_Set1)
disp('Example of evaluating on Set1 from the Rendered WB dataset');
dataset_name = 'RenderedWB_Set1';
imgin = 'Canon1DsMkIII_0087_F_P.png';
in_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set1', ...
    'input');
gt_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set1', ...
    'groundtruth');
metadata_base = fullfile('..', 'examples_from_datasets', ...
    'RenderedWB_Set1', 'metadata');
fprintf('Reading image: %s\n',imgin);
I_in = im2double(imread(fullfile(in_base,imgin))); % read input image
metadata = get_metadata(imgin, dataset_name, metadata_base); % get metadata
cc_mask = round(metadata.cc_mask); % round any float to nearest integer
cc_mask(cc_mask==0) = cc_mask(cc_mask==0)+1; % to start from 1
gt = imread(fullfile(gt_base,metadata.gt_filename)); % read gt image
% hide the color chart from both images before processing and evaluation
I_in(cc_mask(2):cc_mask(2)+cc_mask(4), ...
    cc_mask(1):cc_mask(1)+cc_mask(3),:)=0;
gt(cc_mask(2):cc_mask(2)+cc_mask(4), ...
    cc_mask(1):cc_mask(1)+cc_mask(3),:)=0;


%% processing (replace this part with your method)
fprintf('Processing image: %s\n',imgin);
I_corr = model.correctImage(I_in); % white balance I_in
I_corr = im2uint8(I_corr); % convert to uint8 image

%% Evaluation 
[deltaE00, MSE, MAE, deltaE76] = ...
    evaluate_cc(I_corr, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n\n\n',...
    deltaE00, MSE, MAE, deltaE76);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Example2 (RenderedWB_Set2)
disp('Example of evaluating on Set2 from the Rendered WB dataset');
dataset_name = 'RenderedWB_Set2';
imgin = 'Mobile_00202.png';
in_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set2', ...
    'input');
gt_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set2', ...
    'groundtruth');
metadata_base = ''; % no metadata directory required for Set2
fprintf('Reading image: %s\n',imgin);
I_in = im2double(imread(fullfile(in_base,imgin))); % read input image
metadata = get_metadata(imgin, dataset_name, metadata_base); % get metadata--just to have a consistent style :-)
gt = imread(fullfile(gt_base,metadata.gt_filename)); % read gt image


%% processing (replace this part with your method)
fprintf('Processing image: %s\n',imgin);
I_corr = model.correctImage(I_in); % white balance I_in
I_corr = im2uint8(I_corr); % convert to uint8 image

%% Evaluation
[deltaE00, MSE, MAE, deltaE76] = ...
    evaluate_cc(I_corr, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n\n\n',...
    deltaE00, MSE, MAE, deltaE76);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Example3 (Rendered_Cube+)
disp('Example of evaluating on the Rendered version of Cube+ dataset');
dataset_name = 'Rendered_Cube+';
imgin = '19_F.JPG';
in_base = fullfile('..', 'examples_from_datasets', 'Rendered_Cube+', ...
    'input');
gt_base = fullfile('..', 'examples_from_datasets', 'Rendered_Cube+', ...
    'groundtruth');
metadata_base = ''; % no metadata directory required for rendered Cube+
fprintf('Reading image: %s\n',imgin);
I_in = im2double(imread(fullfile(in_base,imgin))); % read input image
metadata = get_metadata(imgin, dataset_name, metadata_base); % get metadata (we need the cube area for evaluation)
gt = imread(fullfile(gt_base,metadata.gt_filename)); % read gt image

%% processing (replace this part with your method)
fprintf('Processing image: %s\n',imgin);
I_corr = model.correctImage(I_in); % white balance I_in
I_corr = im2uint8(I_corr); % to uint8 image

%% Evaluation 
[deltaE00, MSE, MAE, deltaE76] = ...
    evaluate_cc(I_corr, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n',...
    deltaE00, MSE, MAE, deltaE76);