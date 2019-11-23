%% Calculate Delta E76 between source and target images.
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
%   -deltaE: the value of Delta E76 between image A and image B.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

function deltaE=calc_deltaE(source,target,color_chart_area)

if size(source,1)*size(source,2)<=color_chart_area
    error('Color chart area should be less than the image area');
end
source=double(rgb2lab(source));
target=double(rgb2lab(target));
source = reshape(source,[],3); %l,a,b
target = reshape(target,[],3); %l,a,b
deltaE = sqrt(sum((source - target).^2,2));
deltaE=sum(deltaE)/(size(deltaE,1)-color_chart_area);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% References:
% [1] http://zschuessler.github.io/DeltaE/learn/