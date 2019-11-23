%% Calculate errors between the corrected image and the ground truth image.
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
%   -corrected: corrected image.
%   -gt: ground-truth image.
%   -color_chart_area: If there is a color chart in the image, that is
%   masked out from both images, this variable represents the number of
%   pixels of the color chart. 
%   -opt: determines the required error metric(s) to be reported. 
%         Options: 
%           opt = 1 delta E 2000 (default).
%           opt = 2 delta E 2000 and mean squared error (MSE)
%           opt = 3 delta E 2000, MSE, and mean angular eror (MAE)
%           opt = 4 delta E 2000, MSE, MAE, and delta E 76
%
% Output:
%   -varargout: a cell contains error between corrected and gt images.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

function varargout = evaluate_cc(corrected, gt, color_chart_area, opt)

if nargin == 3
    opt = 1;
end
switch opt
    case 1
        varargout{1} = calc_deltaE2000(corrected,gt,color_chart_area);
    case 2
        varargout{1} = calc_deltaE2000(corrected,gt,color_chart_area);
        varargout{2} =calc_mse(corrected,gt,color_chart_area);
    case 3
        varargout{1} = calc_deltaE2000(corrected,gt,color_chart_area);
        varargout{2} =calc_mse(corrected,gt,color_chart_area);
        varargout{3} =calc_mae(reshape(corrected,[],3),reshape(gt,[],3),...
            color_chart_area);
    case 4
        varargout{1} = calc_deltaE2000(corrected,gt,color_chart_area);
        varargout{2} =calc_mse(corrected,gt,color_chart_area);
        varargout{3} =calc_mae(reshape(corrected,[],3),reshape(gt,[],3),...
            color_chart_area);
        varargout{4} = calc_deltaE(corrected,gt,color_chart_area);
    otherwise
        error('Error in evaluate_cc function');
end
end