clear all
close all



load('ctrl_trajs.mat')

figure('position',[500,500,460,300]);


    minR=0.5;
    maxR=9.5;
    nBasis=9;
    smoothwindow=1; % 1 = no smoothing
   
 
   % knots  = linspace(minR,maxR, 5);
    %dx = diff(knots);
   % dx = dx(1);
   % eps = dx*1.1;
    
    
    bootstraps=250;
    bootstrapped_sols = zeros(bootstraps, nBasis);
    
   % move_only = find(initdists>6);
   % chrom_positions = chrom_positions(move_only);
   % pole1_positions = pole1_positions(move_only);
  %  pole2_positions = pole2_positions(move_only);

    nTrajs = length(chrom_positions);
    
    errboots = zeros(bootstraps,1);


    
    parfor b=1:bootstraps
        bootstrapped_indicies = randsample(nTrajs, nTrajs,true);
        
        chrom_pos_boot ={};
        pole1_boot ={};
        pole2_boot ={};
        
        for k=1:nTrajs
            chrom_pos_boot{k} = chrom_positions{bootstrapped_indicies(k)};
            pole1_boot{k} = pole1_positions{bootstrapped_indicies(k)};
            pole2_boot{k} = pole2_positions{bootstrapped_indicies(k)};
        end
        %rbf_boot = do_model2_rbf(chrom_pos_boot,pole1_boot,pole2_boot,minR,maxR,nBasis,eps,smoothwindow);
            cheby_boot = do_model1(chrom_pos_boot,pole1_boot,pole2_boot,minR,maxR,nBasis,smoothwindow);

        bootstrapped_sols(b,:) = cheby_boot;
    end
    
    %% do some plotting stuff
    
    mToPlot = 111;
    vals_to_plot = linspace(minR,maxR, mToPlot);
    cheb_b_vals = zeros(mToPlot, bootstraps);
    parfor b=1:bootstraps
        %rbfevals_b = rbf_interp ( knots, eps, @phi1, bootstrapped_sols(b,:)', vals_to_plot ); %t_project_value_ab ( mToPlot, nBasis-1, vals_to_plot, bootstrapped_sols(b,:), minR, maxR );
        %rbf_b_vals(:,b) = rbfevals_b;
            chebyevals_b = t_project_value_ab ( mToPlot, nBasis-1, vals_to_plot, bootstrapped_sols(b,:), minR, maxR );
    cheb_b_vals(:,b) = 60*chebyevals_b/5;
        %plot(vals_to_plot,chebyevals_b,'LineWidth',0.25,'color',[.8 .8 .8]); hold on;
    end
    
    
    lvl1=.1;
    lvl2=.05;
    lvl3=.01;
    
    Y1=quantile(cheb_b_vals',[lvl1 1-lvl1;]);
    Y2=quantile(cheb_b_vals',[lvl2 1-lvl2;]);
    Y3=quantile(cheb_b_vals',[lvl3 1-lvl3;]);
    
    
    
    lo_vals1 = mean(cheb_b_vals,2)-std(cheb_b_vals,0,2);
    hi_vals1 =  mean(cheb_b_vals,2)+std(cheb_b_vals,0,2);
    lo_vals2 = mean(cheb_b_vals,2)-2*std(cheb_b_vals,0,2);
    hi_vals2 =  mean(cheb_b_vals,2)+2*std(cheb_b_vals,0,2);
    lo_vals3 = mean(cheb_b_vals,2)-3*std(cheb_b_vals,0,2);
    hi_vals3 =  mean(cheb_b_vals,2)+3*std(cheb_b_vals,0,2);
    
    
   %    patch([vals_to_plot,fliplr(vals_to_plot)], [lo_vals3;flipud(hi_vals3)],[.6 .6 .6],'FaceAlpha',0.25, 'EdgeColor','none','HandleVisibility','off');
  %   hold on;
   %    patch([vals_to_plot,fliplr(vals_to_plot)], [lo_vals2;flipud(hi_vals2)],[.6 .6 .6],'FaceAlpha',0.25, 'EdgeColor','none','HandleVisibility','off');
    %   patch([vals_to_plot,fliplr(vals_to_plot)], [lo_vals1;flipud(hi_vals1)],[.6 .6 .6],'FaceAlpha',0.25, 'EdgeColor','none','HandleVisibility','off');
   % hold on;
    
    patch([vals_to_plot,fliplr(vals_to_plot)], [Y3(1,:)';flipud(Y3(2,:)')],[.6 .6 .6],'FaceAlpha',0.15, 'EdgeColor','none','HandleVisibility','off');
 hold on;
    % hold on;
    %  patch([vals_to_plot,fliplr(vals_to_plot)], [Y2(1,:)';flipud(Y2(2,:)')],[.6 .6 .6],'FaceAlpha',0.15, 'EdgeColor','none','HandleVisibility','off');
    %  patch([vals_to_plot,fliplr(vals_to_plot)], [Y1(1,:)';flipud(Y1(2,:)')],[.6 .6 .6],'FaceAlpha',0.15, 'EdgeColor','none','HandleVisibility','off');
    
    
    
    % plot(vals_to_plot,mean(rbf_b_vals,2),'LineWidth',1,'color','r');
    
   % [solved_rbf_true] = do_model2_rbf(chrom_positions,pole1_positions,pole2_positions,minR,maxR,nBasis,eps,smoothwindow);
    
   % rbf_true = rbf_interp ( knots, eps, @phi1, solved_rbf_true, vals_to_plot );% t_project_value_ab ( mToPlot, nBasis-1, vals_to_plot, solved_cheby_true, minR, maxR );
    
    [solved_cheby_true] = do_model1(chrom_positions,pole1_positions,pole2_positions,minR,maxR,nBasis,smoothwindow);
    
    chebyevals_true = t_project_value_ab ( mToPlot, nBasis-1, vals_to_plot, solved_cheby_true, minR, maxR );


    fun = @(x,xdata)x(1)-x(2).*exp(-xdata./x(3));
x0 = [.04 .1 2];
%x = lsqcurvefit(fun,x0,vals_to_plot,chebyevals_true')


plot(vals_to_plot,60*chebyevals_true/5,'LineWidth',2);
hold on;
%plot(vals_to_plot,fun(x,vals_to_plot));


    hold on;
    
    axis tight;
    fig = gcf;
    fig.Color = 'white';
    fig.InvertHardcopy = 'off';
    
    set(gcf, 'color', 'w');
    set(gca, 'color', 'w');
    pbaspect([1.15,1,1])

    set(gcf, 'InvertHardcopy', 'off');
ylim([-1 4])
xlim([1 8])
yticks([-1 0 1 2 3 4]);
    hold on;
    pbaspect([1.15,1,1])

set(gca,'TickDir','out')