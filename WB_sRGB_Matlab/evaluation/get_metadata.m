%% Get the metadata of the given image fileName
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
% Input:
%   -fileName: image fileName
%   -set: 'Set1', 'Set2', or 'Cube+' (note that we did not include
%   MIT-Adobe5K here, as there are no ground-truth white-balanced images for
%   this set.
%   -base: the metadata directory (except for Set1, use base='';
%
% Output:
%   -data: the metadata of the given image fileName based on the opt value.
%
%  evaluation_examples.m provides some examples of how to use it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

function [ data ] = get_metadata(fileName, set, base)
if nargin==2
    base ='';
end
[~, name, ext] = fileparts(fileName); % get file parts
switch set
    case 'RenderedWB_Set1' % Rendered WB dataset (Set1)
        metadatafile_color=[name,'_color.txt']; % chart's colors info
        metadatafile_mask=[name,'_mask.txt']; % chart coordinates info
        % get color info
        fileId=fopen(fullfile(base,metadatafile_color),'r');
        C=textscan(fileId,'%s\n'); C=C{1}; fclose(fileId);
        colors=zeros(3,24); % 3 x 24 colors in the color chart
        for i=1:24
            temp=strsplit(C{i},',');
            for m=1:3
                colors(m,i)=str2double(temp{m});
            end
        end
        % get coordinates info
        fileId=fopen(fullfile(base,metadatafile_mask),'r');
        C=textscan(fileId,'%s\n'); temp=C{1}; fclose(fileId);
        temp=strsplit(temp{1},',');
        mask=zeros(4,1);
        for m=1:4
            mask(m)=str2double(temp{m});
        end
        % get ground truth file name (without gt directory)
        temp=strsplit(name,'_');
        gt_file='';
        for m=1:length(temp)-2
            gt_file=strcat(gt_file,temp{m},'_');
        end
        gt_file=[gt_file,'G_AS.png'];
        % computes mask area
        mask_area = mask(3) * mask(4);
        % final metadata
        data.gt_filename=gt_file;
        data.cc_colors=colors;
        data.cc_mask=mask;
        data.cc_mask_area=mask_area;
        
    case 'RenderedWB_Set2' % Rendered WB dataset (Set2)
        data.gt_filename=[name, ext];
        data.cc_colors=[];
        data.cc_mask=[];
        data.cc_mask_area=0;
        
    case 'Rendered_Cube+' % Rendered Cube+
        %  get ground-truth filename
        [~,name,ext] = fileparts(fileName);
        parts = strsplit(name,'_');
        data.gt_filename=[parts{1} ext];
        data.cc_colors=[]; % no chart's colors
        data.cc_mask=[]; % we already masked out it during rendering
        data.cc_mask_area=58373; % calibration obj's area is fixed over all images
    otherwise
        error('Invalid value for set variable. Please use: ''RenderedWB_Set1'', ''RenderedWB_Set2'', ''Rendered_Cube+''');
end
end