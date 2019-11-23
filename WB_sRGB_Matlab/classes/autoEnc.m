%% autoEncoder class
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

classdef autoEnc
    properties
        weights
        bias
    end
    methods
        function feature = encode(obj,hist)
            feature = (obj.sigmf(obj.weights * reshape(hist,[],1) + obj.bias,[1,0]))';
        end
        function y = sigmf(obj, x, params)
            a = cast(params(1),'like',x);
            c = cast(params(2),'like',x);
            y = 1./(1 + exp(-a*(x-c)));
            
        end
    end
end