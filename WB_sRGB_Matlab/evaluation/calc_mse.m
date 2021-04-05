%% Calculate mean squared error between source and target images.
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
%   -source: image A
%   -target: image B 
%   -color_chart_area: If there is a color chart in the image (that is
%   masked out from both images, this variable represents the number of
%   pixels of the color chart.
%
% Output:
%   -f: the mean squared error between image A and image B.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

function mse=calc_mse(source,target,color_chart_area)

if size(source,1)*size(source,2)<=color_chart_area
    error('Color chart area should be less than the image area');
end
source=double(source);
target=double(target);
diff=(source(:)-target(:)).^2;
mse=sum(diff)/(length(diff)-color_chart_area); % yes, color_chart_area x 3 makes more sense :-), but we used this (without x 3) to evaluate all methods
end
