addpath('classes');

infileName = fullfile('a0028-_DSC0032_AS.JPG');

outfileName = fullfile('result.jpg');

load('model_gpu.mat');

I_in = gpuArray(imread(infileName));

tic
I_corr = model.correctImage(I_in);
toc

I_corr = gather(I_corr);

subplot(1,2,1); imshow(I_in); title('Input');

subplot(1,2,2); imshow(I_corr); title('Our result');

imwrite(I_corr,outfileName);