%% GUI Demo: White balancing a single image
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


function varargout = demo_GUI(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @demo_GUI_OpeningFcn, ...
    'gui_OutputFcn',  @demo_GUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function demo_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
global model
global gpu
global upgraded
handles.output = hObject;
guidata(hObject, handles);
handles.save.Enable = 'off';
upgraded = handles.upgradedModel.Value;
gpu = handles.device.Value;
if upgraded == 1 && gpu == 1
    load(fullfile('models','WB_model+_GPU.mat'));
elseif gpu == 1
    load(fullfile('models','WB_model_GPU.mat'));
elseif upgraded == 1
    load(fullfile('models','WB_model+.mat'));
else
    load(fullfile('models','WB_model.mat'));
end
handles.k.Value=model.K;

function varargout = demo_GUI_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

function browse_Callback(hObject, eventdata, handles)
global Path_Name
global File_Name
global I
global I_corr
global model
global feature
global hist
global shownImage
Old_fileName = File_Name;
Old_pathName = Path_Name;
[File_Name, Path_Name] = uigetfile({'*.jpg';'*.png';'*.jpeg'},...
    'Select input image',fullfile(pwd,'..','example_images'));


if File_Name == 0
    File_Name = Old_fileName;
    Path_Name = Old_pathName;
    if sum(I(:)) ~= 0
        handles.save.Enable = 'on';
    else
        handles.save.Enable = 'off';
    end
else
    I = imread(fullfile(Path_Name,File_Name));
    axes(handles.image);
    imshow(I);
    pause(0.001);
    handles.status.String = 'Processing...';pause(0.001);
    hist = model.RGB_UVhist(im2double(I));
    feature = model.encode(hist);
    k = handles.k.Value;
    sigma = handles.sigma.Value;
    model.K = round(k);
    model.gamut_mapping = handles.clipping.Value + 1;
    I_corr = model.correctImage(I,feature,sigma);
    imshow(I_corr);
    handles.save.Enable = 'on';
    shownImage = 'corrected';
    handles.status.String = 'Done!';
    pause(0.01); handles.status.String = '';
end



function save_Callback(hObject, eventdata, handles)
global Path_Name
global File_Name
global I_corr
global gpu
[~,name,ext] = fileparts(File_Name);
outFile_Name = [name '_WB' ext];
[file,path,~] = uiputfile({'*.jpg';'*.png';'*.jpeg';'*.*'},'Save Image',...
    fullfile(Path_Name,outFile_Name));
if file ~=0
    handles.status.String = 'Processing...';pause(0.001);
    if gpu == 1
        imwrite(gather(I_corr), fullfile(path,file));
    else
        imwrite(I_corr, fullfile(path,file));
    end
    handles.status.String = 'Done!';
    pause(0.01); handles.status.String = '';
end


function k_Callback(hObject, eventdata, handles)
global model
global I
global feature
global I_corr
axes(handles.image);
handles.status.String = 'Processing...';pause(0.001);
k = handles.k.Value;
sigma = handles.sigma.Value;
model.K = round(k);
model.gamut_mapping = handles.clipping.Value + 1;
I_corr = model.correctImage(I,feature,sigma);
imshow(I_corr);
handles.status.String = 'Done!';
pause(0.01); handles.status.String = '';



function k_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,...
        'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function sigma_Callback(hObject, eventdata, handles)
global model
global I
global feature
global I_corr
axes(handles.image);
handles.status.String = 'Processing...';pause(0.001);
k = handles.k.Value;
sigma = handles.sigma.Value;
model.K = round(k);
model.gamut_mapping = handles.clipping.Value + 1;
I_corr = model.correctImage(I,feature,sigma);
imshow(I_corr);
handles.status.String = 'Done!';
pause(0.01); handles.status.String = '';


function sigma_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,...
        'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function originalImg_Callback(hObject, eventdata, handles)
global shownImage
global I
global I_corr
axes(handles.image);
if strcmpi(shownImage,'original')
    imshow(I_corr);
    shownImage = 'corrected';
else
    imshow(I);
    shownImage = 'original';
end


function device_Callback(hObject, eventdata, handles)
global upgraded
global gpu
global model
global I
global feature
global I_corr
upgraded = handles.upgradedModel.Value;
gpu = handles.device.Value;

if gpu == 1 
    try
        gpuDevice();
    catch
        error('Cannot find a GPU device');
    end
end
if upgraded == 1 && gpu == 1
    load(fullfile('models','WB_model+_GPU.mat'));
elseif gpu == 1
    load(fullfile('models','WB_model_GPU.mat'));
elseif upgraded == 1
    load(fullfile('models','WB_model+.mat'));
else
    load(fullfile('models','WB_model.mat'));
end

if isempty(I) == 0
    axes(handles.image);
    handles.status.String = 'Processing...';pause(0.001);
    k = handles.k.Value;
    sigma = handles.sigma.Value;
    model.K = round(k);
    model.gamut_mapping = handles.clipping.Value + 1;
    I_corr = model.correctImage(I,feature,sigma);
    imshow(I_corr);
    handles.status.String = 'Done!';
    pause(0.01); handles.status.String = '';
end

function upgradedModel_Callback(hObject, eventdata, handles)
global upgraded
global gpu
global model
global I
global hist
global feature
global I_corr
upgraded = handles.upgradedModel.Value;
gpu = handles.device.Value;
if upgraded == 1 && gpu == 1
    load(fullfile('models','WB_model+_GPU.mat'));
elseif gpu == 1
    load(fullfile('models','WB_model_GPU.mat'));
elseif upgraded == 1
    load(fullfile('models','WB_model+.mat'));
else
    load(fullfile('models','WB_model.mat'));
end

feature = model.encode(hist);
handles.k.Value = model.K;

if isempty(I) == 0
    axes(handles.image);
    handles.status.String = 'Processing...';pause(0.001);
    k = handles.k.Value;
    sigma = handles.sigma.Value;
    model.K = round(k);
    model.gamut_mapping = handles.clipping.Value + 1;
    I_corr = model.correctImage(I,feature,sigma);
    imshow(I_corr);
    handles.status.String = 'Done!';
    pause(0.01); handles.status.String = '';
end

function clipping_Callback(hObject, eventdata, handles)
global model
global I
global feature
handles.scaling.Value = ~handles.clipping.Value;
if isempty(I) == 0
    axes(handles.image);
    handles.status.String = 'Processing...';pause(0.001);
    k = handles.k.Value;
    sigma = handles.sigma.Value;
    model.K = round(k);
    model.gamut_mapping = handles.clipping.Value + 1;
    I_corr = model.correctImage(I,feature,sigma);
    imshow(I_corr);
end
handles.status.String = 'Done!';
pause(0.01); handles.status.String = '';


function scaling_Callback(hObject, eventdata, handles)
global model
global I
global feature
handles.clipping.Value = ~handles.scaling.Value;
if isempty(I) == 0
    axes(handles.image);
    handles.status.String = 'Processing...';pause(0.001);
    k = handles.k.Value;
    sigma = handles.sigma.Value;
    model.K = round(k);
	model.gamut_mapping = handles.clipping.Value + 1;
    I_corr = model.correctImage(I,feature,sigma);
    imshow(I_corr);
end
handles.status.String = 'Done!';
pause(0.01); handles.status.String = '';
