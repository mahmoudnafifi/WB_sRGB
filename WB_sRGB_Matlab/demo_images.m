%% Demo: White balancing all images in a directory
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

%% input and options
in_image_dir = fullfile('..', 'example_images');
out_image_dir = fullfile('..', 'example_images_WB');
device = 'cpu'; %'cpu' or 'gpu'
gamut_mapping = 2; % use 1 for scaling, 2 for clipping (our paper's results
% reported using clipping). If the image is over-saturated, scaling is
% recommended.
upgraded_model = 0; % use 1 to load our new model that is upgraded with new
% training examples.

%% 
switch lower(device)
    case 'cpu'
        if upgraded_model == 1
            load(fullfile('models','WB_model+.mat'));
        elseif upgraded_model == 0
            load(fullfile('models','WB_model.mat'));
        else
            error('Wrong upgraded_model value; please use 0 or 1');
        end
    case 'gpu'
        try
            gpuDevice();
        catch
            error('Cannot find a GPU device');
        end
        if upgraded_model == 1
            load(fullfile('models','WB_model+_gpu.mat'));
        elseif upgraded_model == 0
            load(fullfile('models','WB_model_gpu.mat'));
        else
            error('Wrong upgraded_model value; please use 0 or 1');
        end
    otherwise
        error('Wrong device; please use ''gpu'' or ''cpu''')
end
model.gamut_mapping = gamut_mapping;
if exist(out_image_dir,'dir')==0
    mkdir(out_image_dir);
end
imds = imageDatastore(in_image_dir);
files = imds.Files;
for f = 1 : length(files)
    fprintf('Processing image: %s\n',files{f});
    infileName = files{f};
    [~,name,ext] = fileparts(infileName);
    outfileName = fullfile(out_image_dir, [name, '_', 'WB', ext]);
    I_in = imread(infileName);
    I_corr = model.correctImage(I_in);
    if strcmpi(device,'gpu')
        imwrite(gather(I_corr),outfileName);
    else
        imwrite(I_corr,outfileName);
    end
    disp('Done!');
end
