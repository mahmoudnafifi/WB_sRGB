%% Start parallel pool using the given number of cores (requires Parallel Computing Toolbox)
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
%   -NUM_WORKERS: number of cores.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

function  startpooling(NUM_WORKERS)
po = gcp('nocreate');
if ~isempty(po)
    if po.NumWorkers ~= NUM_WORKERS
        delete(po);
        pc=parcluster('local');
        pc.NumWorkers=NUM_WORKERS;
        po = parpool(NUM_WORKERS);
    end
else
    pc=parcluster('local');
    pc.NumWorkers=NUM_WORKERS;
    po = parpool(NUM_WORKERS);
end
end