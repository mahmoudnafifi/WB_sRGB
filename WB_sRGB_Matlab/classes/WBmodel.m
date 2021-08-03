%% White-balance model class
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

classdef WBmodel
    properties
        features % training features
        mappingFuncs % training mapping functions
        K % K value for KNN
        encoder % autoEnc or PCA object
        gamut_mapping % mapping inside the gamut (=1 for scaling, =2 for 
        % clipping). Our results reported using clipping; however, we found
        % scaling gives compelling results with over-saturated eexamples
    end
    methods
        function feature = encode(obj,hist) 
            % Generates a compacted feature of a given RGB-uv histogram
            % tensor. %
            feature =  obj.encoder.encode(hist);
        end
        
        function hist = RGB_UVhist(obj,I)  
            % Computes an RGB-uv histogram tensor. %
            I = im2double(I);
            if size(I,1)*size(I,2) > 202500 % if it is larger than 450*450
                factor = sqrt(202500/(size(I,1)*size(I,2))); % rescale factor
                newH = floor(size(I,1)*factor); % new height 
                newW = floor(size(I,2)*factor); % new width 
                I = imresize(I,[newH,newW],'nearest'); % resize image
            end
            
            h= sqrt(max(size(obj.encoder.weights,1),...
                size(obj.encoder.weights,2))/3);
            eps= 6.4/h; 
            I=(reshape(I,[],3));
            A=[-3.2:eps:3.19]; % dummy vector
            hist=zeros(size(A,2),size(A,2),3); % histogram will be stored here
            i_ind=I(:,1)~=0 & I(:,2)~=0 & I(:,3)~=0; 
            I=I(i_ind,:); % remove zereo pixels
            Iy=sqrt(I(:,1).^2+I(:,2).^2+I(:,3).^2); % intensity vector
            for i = 1 : 3 % for each color channel, do 
                r = setdiff([1,2,3],i); % exclude the current color channel
                Iu=log((I(:,i))./(I(:,r(1)))); % current color channel / the first excluded channel
                Iv=log((I(:,i))./(I(:,r(2)))); % current color channel / the second excluded channel
                diff_u=abs(Iu-A); % differences in u space
                diff_v=abs(Iv-A); % differences in v space
                % for old Matlab versions:
                % diff_u=abs(repmat(Iu,[1,size(A,2)])-repmat(A,[size(Iu,1),1]));
                % diff_v=abs(repmat(Iv,[1,size(A,2)])-repmat(A,[size(Iv,1),1]));
                % here, we will use a matrix multiplication expression to compute eq. 4 in the main paper.
                diff_u=(reshape((reshape(diff_u,[],1)<=eps/2),...
                    [],size(A,2))); % don't count any pixel has difference beyond the threshold in the u space
                diff_v=(reshape((reshape(diff_v,[],1)<=eps/2),...
                    [],size(A,2))); % similar in the v space
                hist(:,:,i)=(Iy.*double(diff_u))'*double(diff_v); % compute the histogram 
                % for old Matlab versions: 
                % hist(:,:,i)=(repmat(Iy, [1, size(diff_u,2)]).* double(diff_u))'*double(diff_v); % compute the histogram 
                hist(:,:,i)=sqrt(hist(:,:,i)/sum(sum(hist(:,:,i)))); % sqrt the histogram after normalizing
            end
            hist = imresize(hist,[h h],'bilinear');
        end
        
        function [corrected, mf, in_gamut] =...
                correctImage (obj,I,feature, sigma) 
            % White balance a given image I %
            I = im2double(I);
            if nargin == 2
                feature = obj.encode(obj.RGB_UVhist(I)); % if no feature is given, compute it
                sigma = 0.25; % fall-off factor for KNN blending
            elseif nargin == 3
                sigma = 0.25;
            end
            M = size(obj.mappingFuncs,2);            
            [dH,idH] = pdist2(obj.features,feature,...
                'euclidean','Smallest',obj.K); % gets naerest K faetures
            weightsH = exp(-((dH).^2)/(2*sigma^2)); % computes weights
            weightsH = weightsH/sum(weightsH); % normalizes weights
            mf = sum(weightsH .* obj.mappingFuncs(idH,:),1); % blends nearest mapping funcs
            % for old Matlab versions:
            % mf = sum(repmat(weightsH,[1, size(obj.mappingFuncs,2)]) .* obj.mappingFuncs(idH,:),1);
            mf = reshape(mf,[M/3,3]); % reshape to be 11x3
            [corrected, in_gamut] = obj.color_correction(I, mf, ...
                obj.gamut_mapping); % correct image's colors
            corrected = double(corrected);
            
        end
        
        function [out,map] = color_correction(obj,input, m, gamut_map)
            % Applies a mapping function m to a given input image. %
            if nargin == 3
                gamut_map = 2;
                map = [];
            end
            sz=size(input);
            input=reshape(input,[],3);
            if gamut_map == 1
                input_ = input; % take a copy--will be used later
            end
            input=obj.kernelP(input); % raise it to a higher degree (Nx11)
            out=input * m; 
            if gamut_map == 1 % if scaling,
                out = obj.norm_scaling(input_, out);
                map = [];
            elseif gamut_map == 2 % if clipping,
                [out,map] = obj.out_of_gamut_clipping(out);
            end
            out=reshape(out,[sz(1),sz(2),sz(3)]); % reshape it from Nx3 to the original size
        end
        
        
        function [I,map] = out_of_gamut_clipping(obj,I)
            % Clips out-of-gamut pixels. %
            I = im2double(I);
            map = ones(size(I)); % in-gamut map
            map(I>1) = 0;
            map(I<0) = 0;
            map = map(:,1) & map(:,2) & map(:,3);
            I(I>1) = 1; % any pixel is higher than 1, clip it to 1
            I(I<0) = 0; % any pixel is below 0, clip it to 0
        end
        
        function [I_corr] = norm_scaling(obj, I, I_corr)
            % Scales each pixel based on original image energy. %
            norm_I_corr = sqrt(sum(I_corr.^2,2));
            inds = norm_I_corr ~= 0;
            norm_I_corr = norm_I_corr(inds);
            norm_I = sqrt(sum(I(inds,:).^2,2));
            I_corr(inds, :) = I_corr(inds,:)./norm_I_corr .* norm_I;
        end

        function O=kernelP(obj,I)
            % kernel(R,G,B)=[R,G,B,RG,RB,GB,R2,G2,B2,RGB,1];
            % Kernel func reference:
            % Hong, et al., "A study of digital camera colorimetric 
            % characterization based on polynomial modeling." Color 
            % Research & Application, 2001.
            O=[I,... %r,g,b
                I(:,1).*I(:,2),I(:,1).*I(:,3),I(:,2).*I(:,3),... %rg,rb,gb
                I.*I,... %r2,g2,b2
                I(:,1).*I(:,2).*I(:,3),... %rgb
                ones(size(I,1),1)]; %1
        end  
    end
end
