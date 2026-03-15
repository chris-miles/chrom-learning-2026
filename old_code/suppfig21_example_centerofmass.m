clear all
close all

load("ctrl_clouds.mat")

addpath(genpath('cbrewer'))

c=9; % or 9

chroms=chrom_clouds{c};
  p1=p1_trajs{c};
  p2=p2_trajs{c};
  endsep=end_seps{c}+15;
  p_cent = p_centers{c};
  chrom_cent=chrom_centers{c};
  chrom_cent_shift = chrom_cent-nanmean(chrom_cent);
  p1_shift = p1-nanmean(chrom_cent);
  p2_shift=p2-nanmean(chrom_cent);
  p_cent_shift = p_cent-nanmean(chrom_cent);
  chrom_cloud = chrom_clouds{c};%-nanmean(chrom_cent);
  chrom_cloud_shift = zeros(size(chrom_cloud));
  for t=1:181
      chrom_cloud_shift(t,:,:)=squeeze(chrom_cloud(t,:,:))-nanmean(chrom_cent);
  end 

   [coeff, chrom_cent_pca, latent, tsquared, explained]=pca(chrom_cent);
   
  %test= score*coeff';
  p1_pca= p1_shift*coeff;
  p2_pca= p2_shift*coeff;
  p_cent_pca = p_cent_shift*coeff;
  chrom_cloud_rot = zeros(size(chrom_cloud));

  for t=1:181
      chrom_cloud_rot(t,:,:)=squeeze(chrom_cloud_shift(t,:,:))*coeff;
  end 
  figure;

set(gcf, 'defaultFigureRenderer', 'painters')

  gray_colormap=cbrewer('seq','Greys',1000,'pchip');
  red_colormap=cbrewer('seq','YlOrRd',1000,'pchip');
  blue_colormap=cbrewer('seq','YlGnBu',1000,'pchip');
  purd_colormap=cbrewer('seq','Purples',1000,'pchip');
  bupu_colormap=cbrewer('seq','Purples',1000,'pchip');
  purple_colormap=cbrewer('seq','Purples',1000,'pchip');


  % truncate because early values are too light
 gray_colormap = gray_colormap(300:end,:);
 red_colormap = red_colormap(100:end,:);
 blue_colormap = blue_colormap(300:end,:);
 
 purd_colormap = purd_colormap(300:end,:);
 bupu_colormap = bupu_colormap(300:end,:);
% green_colormap = green_colormap(1:500,:);
 purple_colormap = purple_colormap(100:800,:);

hold on
 for j = 1:46
    this_chrom = squeeze(chrom_cloud_rot(1:endsep,j,:));
colorline_2d(this_chrom(:,1),this_chrom(:,2),...
    1:endsep,purple_colormap,.25,0);    
 end 

 
colorline_2d(p1_pca(1:endsep,1),p1_pca(1:endsep,2),...
    1:endsep, gray_colormap,4,1);
hold on;
colorline_2d(p2_pca(1:endsep,1),p2_pca(1:endsep,2),...
    1:endsep, gray_colormap,4,1);

colorline_2d(p_cent_pca(1:endsep,1),p_cent_pca(1:endsep,2),...
    1:endsep,blue_colormap,4,1)

colorline_2d(chrom_cent_pca(1:endsep,1),chrom_cent_pca(1:endsep,2),...
    1:endsep,red_colormap,4,1);

axis tight;
axis equal;
set(gca,'TickDir','out')

grid;
set(gcf, 'defaultFigureRenderer', 'painters')
axis equal;
ylim([-4 4]);
xlim([-10 10])
xticks(-10:2:10)
yticks([-4:2:4])
