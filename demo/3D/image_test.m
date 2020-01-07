% Add paths
addpath('subaxis/')

%% SPAMM IMAGE
clear all; clc;
load('tagging_image_0000_0000.mat');
Is = I;
load('dense_image_0000_0000.mat');
Id = I;
load('tvm_image_0000_0000.mat');
It = I;
clearvars I;

% Range SPAMM for plot
maxs = max(Is(:));
mins = 1.3*min(Is(:));
sxs = 50;
sys = 50;
szs = 15;

% Displacement magnitude;
% I = sqrt(I(:,:,:,1,:).^2 + I(:,:,:,2,:).^2 + I(:,:,:,3,:).^2);
I1 = It(:,:,:,1,:);
I2 = It(:,:,:,2,:);
I3 = It(:,:,:,3,:);
Mt = sqrt(I1.^2 + I2.^2 + I3.^2);
s = size(Mt)

% Range for plot
maxt = 0.8*max(Mt(:));
mint = min(Mt(:));
sxt = 20;
syt = 20;
szt = 15;

% Displacement magnitude;
% I = sqrt(I(:,:,:,1,:).^2 + I(:,:,:,2,:).^2 + I(:,:,:,3,:).^2);
I1 = squeeze(Id(:,:,:,1,:));
I2 = squeeze(Id(:,:,:,2,:));
I3 = squeeze(Id(:,:,:,3,:));
Md = sqrt(I1.^2 + I2.^2 + I3.^2);
s = size(Md);

% Range for DENSE plot
maxd = max(I1(:));
mind = min(I1(:));
sxd = 20;
syd = 20;
szd = 15;


for i=1:20
    figure('visible','off')
    subaxis(2, 3, 1, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
    imagesc(Is(:,:,szs,i), 'XData', [-0.0875,0.0875], 'YData', [-0.0875,0.0875]);
    colormap gray; axis equal off; caxis([mins, maxs])
    subaxis(2, 3, 2, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
    imagesc(I1(:,:,szd,i), 'XData', [-0.0875,0.0875], 'YData', [-0.0875,0.0875]);
    colormap gray; axis equal off; caxis([mind, maxd])
    subaxis(2, 3, 3, 'sh', 0.03, 'sv', 0.0000001, 'padding', 0, 'margin', 0);
    imagesc(Mt(:,:,szt,i), 'XData', [-0.0875,0.0875], 'YData', [-0.0875,0.0875]);
    colormap gray; axis equal off; caxis([mint, maxt])    

    subaxis(2, 3, 4, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
    imagesc(squeeze(Is(sxs,:,end:-1:1,i))', 'XData', [-0.0875,0.0875], 'YData', [-0.1125,0.1125]);
    colormap gray; axis equal off; caxis([mins, maxs])
    xlabel('SPAMM', 'interpreter', 'LaTeX')
    subaxis(2, 3, 5, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
    imagesc(squeeze(I3(sxd,:,end:-1:1,i))', 'XData', [-0.0875,0.0875], 'YData', [-0.1125,0.1125]);
    colormap gray; axis equal off; caxis([mind, maxd])    
    xlabel('DENSE', 'interpreter', 'LaTeX')
    subaxis(2, 3, 6, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
    imagesc(squeeze(Mt(sxt,:,end:-1:1,i))', 'XData', [-0.0875,0.0875], 'YData', [-0.1125,0.1125]);
    colormap gray; axis equal off; caxis([mint, maxt])    
    xlabel('TVM', 'interpreter', 'LaTeX')
    
    print(['spamm_',sprintf('%04d', i)],'-dpng')
    pause(0.01)
end

% for i=1:30
%     image_test
% end

% figure,
% sizeIn = size(I1)
% slice(double(I1(:,:,:,15)),sizeIn(2)/2,sizeIn(1)/2,15);
% grid on, shading interp, colormap gray

% %% TVM IMAGE
% clear all; clc;
% load('tvm_image_0000_0000.mat');
% 
% % Displacement magnitude;
% % I = sqrt(I(:,:,:,1,:).^2 + I(:,:,:,2,:).^2 + I(:,:,:,3,:).^2);
% I1 = I(:,:,:,1,:);
% I2 = I(:,:,:,2,:);
% I3 = I(:,:,:,3,:);
% M = sqrt(I1.^2 + I2.^2 + I3.^2);
% s = size(M)
% 
% % Range for plot
% max = 0.8*max(M(:));
% min = min(M(:));
% sx = 20;
% sy = 20;
% sz = 15;
% 
% % figure,
% % sizeIn = size(I1)
% % slice(double(I1(:,:,:,15)),sizeIn(2)/2,sizeIn(1)/2,15);
% % grid on, shading interp, colormap gray
% 
% for i=1:20
%     figure('visible','off')
%     subaxis(1, 2, 1, 'sh', 0.03, 'sv', 0.0000001, 'padding', 0, 'margin', 0);
%     imagesc(M(:,:,sz,i), 'XData', [-0.0875,0.0875], 'YData', [-0.0875,0.0875]); colormap gray; axis equal off; caxis([min, max])
%     subaxis(1, 2, 2, 'sh', 0.03, 'sv', 0.0000001, 'padding', 0, 'margin', 0);
%     imagesc(squeeze(M(sx,:,end:-1:1,i))', 'XData', [-0.0875,0.0875], 'YData', [-0.1125,0.1125]); colormap gray; axis equal off; caxis([min, max])
%     print(['tvm_',sprintf('%04d', i)],'-dpng')
%     pause(0.2)
% end
