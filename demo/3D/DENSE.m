clear all; clc; close all

% Load DENSE image
load I_DENSE.mat
load mask.mat
mask(mask<1e-10) = 0.125;
mask(mask==1.0) = 0;

% Extract magnitude and phase
m = abs(I_DENSE);
u = -angle(I_DENSE);

% Magnitude components
mx = squeeze(m(:,:,:,1,:)); 
my = squeeze(m(:,:,:,2,:)); 
mz = squeeze(m(:,:,:,3,:)); 

% Phase components
Xpha = squeeze(u(:,:,:,1,:));
Ypha = squeeze(u(:,:,:,2,:));
Zpha = squeeze(u(:,:,:,3,:));

%% Plots
figure('Visible', 'off')
slice = 10;
imagesc(Xpha(:,:,slice,10)); set(gca, 'Ydir', 'normal'); hold on
caxis([-pi pi]); colormap gray; axis off equal
green = cat(3, zeros(55,55), ones(55,55), zeros(55,55)); 
h = imagesc(green); hold off 
caxis([-pi pi]); colormap gray; axis off equal;
set(h, 'AlphaData', mask(:,:,slice,10))
export_fig('Xpha','-png')

figure('Visible', 'off')
imagesc(Ypha(:,:,slice,10)); set(gca, 'Ydir', 'normal'); hold on 
green = cat(3, zeros(55,55), ones(55,55), zeros(55,55)); 
h = imagesc(green); hold off 
caxis([-pi pi]); colormap gray; axis off equal;
set(h, 'AlphaData', mask(:,:,slice,10))
export_fig('Ypha','-png')

figure('Visible', 'off')
imagesc(Zpha(:,:,slice,10)); set(gca, 'Ydir', 'normal'); hold on 
green = cat(3, zeros(55,55), ones(55,55), zeros(55,55)); 
h = imagesc(green); hold off 
caxis([-pi pi]); colormap gray; axis off equal;
set(h, 'AlphaData', mask(:,:,slice,10))
export_fig('Zpha','-png')

figure('Visible', 'off')
imagesc(mx(:,:,slice,10)); set(gca, 'Ydir', 'normal');
axis off equal; colormap gray
export_fig('M','-png')