%% Calculate mean angular error between source and target images
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
%   -f: the mean angular error between image A and image B.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
function f = calc_mae(source, target,color_chart_area)

if size(source,1)<=color_chart_area
    error('Color chart area should be less than the image area');
end
source=double(source);
target=double(target);
target_norm = sqrt(sum(target.^2,2));
source_mapped_norm = sqrt(sum(source.^2,2));
angles=dot(source,target,2)./(source_mapped_norm.*target_norm);
angles(angles>1)=1;
f=acosd(angles);
f(isnan(f))=0;
f=sum(f)/(size(source,1)-color_chart_area);
end