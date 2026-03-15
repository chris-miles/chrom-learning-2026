clear all
close all

load('ctrl_trajs_cloud.mat')

    lagmax=101;
    smoothwindow = 31;


 acs = [];
 err_ests=[];
 
nCells=length(chrom_clouds);

for c=1:nCells
  chroms=chrom_clouds{c};
  p1=p1_trajs{c};
  p2=p2_trajs{c};
  endsep=end_seps{c};
  p_cent = p_centers{c};
  chrom_cent=chrom_centers{c};
  



tmax = endsep+10;


times_to_plot = 1:tmax;

chr_cent_smooth = smoothdata(chrom_cent,1,'sgolay',smoothwindow);
p_cent_smooth = smoothdata(p_cent,1,'sgolay',smoothwindow);

p_vel = diff(p_cent_smooth);
chr_cent_vel = diff(chr_cent_smooth);

[ac,err_est] = dot_autocov(p_vel',chr_cent_vel',lagmax);
acs=[acs;ac];
err_ests=[err_ests;err_est];
    
end

    lags = 5*(-lagmax:1:lagmax);
    errbar_cont(lags,nanmedian(acs),nanstd(acs) )
hold on;
%errorbar(lags, nanmean(acs),nanstd(acs),'LineWidth',2,'Capsize',3);

   % plot(lags,nanmean(acs),'LineWidth',3,'color','k')
    hold on;
    plot(lags, acs','LineWidth',0.75);
    xline(0)
    yline(0)
    xlim([-200,200])
    ylim([-.25,1]);
    axis square;
    plot(lags, nanmedian(acs),'k','LineWidth',3);
set(gca,'TickDir','out')
box on;
box off;
%set(gca,'LineWidth',1.5)
pbaspect([1.15 1 1])
yticks([-.25 0 0.25 .5 .75 1])