clear all; clc; close all

%% Load DENSE image
load I_CSPAMM.mat
load mask.mat
mask(mask<1e-10) = 0.125;
mask(mask==1.0) = 0;

% extract image
I_CSPAMM = double(I.magnitude.Image);

% Slice number and cardiac phase
slice = 9;
cp    = 10;

% Extract components
tx = real(squeeze(I_CSPAMM(:,:,slice,1,cp)));
ty = real(squeeze(I_CSPAMM(:,:,slice,2,cp)));

% Image size
imsize = size(tx);

%% Plots
figure('Visible', 'on')
imagesc(tx.*ty); set(gca, 'Ydir', 'normal'); hold on
colormap gray; axis off equal
export_fig('t','-png')

%% Load SPAMM image
load I_SPAMM.mat
load mask.mat
mask(mask<1e-10) = 0.125;
mask(mask==1.0) = 0;

% extract image
I_SPAMM = double(Is.magnitude.Image);

% Slice number and cardiac phase
slice = 9;
cp    = 10;

% Extract components
tx = real(squeeze(I_SPAMM(:,:,slice,1,cp)));
ty = real(squeeze(I_SPAMM(:,:,slice,2,cp)));

% Image size
imsize = size(tx);

%% Plots
figure('Visible', 'on')
imagesc(tx.*ty); set(gca, 'Ydir', 'normal'); hold on
colormap gray; axis off equal
export_fig('ts','-png')