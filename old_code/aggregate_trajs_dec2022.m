clear all
close all


addpath(genpath('CPAnalysis'))

load('CPAnalysis/table/tab_h0_cp.mat')
load('CPAnalysis/table/tab_stat_h0.mat')


ang = @(u,v) atan2(norm(cross(u,v)),dot(u,v));

cell_db

ttt=3;

switch ttt
    case 1
        cellIDs=cellIDs_ctr;
        filename='ctrl_trajs';

    case 2
        cellIDs=cellIDs_cenp;
        filename='gsk_trajs';

    case 3
        cellIDs=cellIDs_dyn;
        filename='rod_trajs';
    case 4
        cellIDs=cellIDs_double;
        filename='gskrod_trajs';
    case 5
        cellIDs=cellIDs_hsp;
        filename='hsp_trajs';
end


num_cells = length(cellIDs);

smoothwindow = 1;
traj_count = 0;
bad_count=0;


    ikt_thresh = .90;
    cTilt_thresh = pi/8;
    cTilt_stable_window = 10;
        window3 = 10; % WAS 8, see if this changes anything
        windowsize = 10;

        dot_thresh = .75;

for c = 1:num_cells
    this_cell = cellIDs{c};
    celltype = strcat('../data/',this_cell,'.mat');



    load(celltype);

    % copy matrices
    cc = centrioles;
    kk = kinetochores;
    cc = cc(1:end,:,:);
    kk = kk(1:end,:,:);

    num_tpts = length(cc); % get number of time pts



    if num_tpts > 181
        num_tpts = 181;
        cc(182:length(cc),:,:) = [];
        kk(182:length(kk),:,:) = [];
    end


    p1_pos_ori    = zeros(num_tpts,3);   % original ppositions of pole 1
    p2_pos_ori    = zeros(num_tpts,3);   % original ppositions of pole 2

    chr_positions = zeros(num_tpts, tracked, 3);

    p1_pos_ori(:,:) = cc(:,1:3,1);
    p2_pos_ori(:,:) = cc(:,1:3,2);


    polepole_cent = 0.5*(p1_pos_ori+p2_pos_ori);

    dist_from_cent = zeros(num_tpts, tracked);

    ikt_dist = zeros(num_tpts,tracked);
    spind_angles = zeros(num_tpts, tracked);


    for i = 1:num_tpts
        kk_pos  = zeros(3,tracked);

        pp_vec = cc(i,1:3,1) - cc(i,1:3,2);
        polepole_dist(i) = vecnorm(pp_vec);



        for j = 1:tracked
            k1 = kk(i,1:3,j);
            k2 = kk(i,4:6,j);
            kkvec = k1-k2;
            kk_pos(:,j) = nanmean([k1;k2],1);
            dist_from_cent(i,j) = vecnorm(kk_pos(:,j)'-polepole_cent(i,:),2,2);
            ikt_dist(i,j) = vecnorm(kkvec);
            spind_angles(i,j) = ang(kkvec,pp_vec);
        end
        chr_positions(i, :,:) = kk_pos';

    end
    %%%%%

    biorient_times = zeros(tracked,1);
    for j =1:tracked

        %% Find when the centromere biorients

        % find when cTilt angle is below threshold and ikt_dist is above
        % specified thresholds
        cTilt_close = zeros(num_tpts,1);
        ikt_close   = zeros(num_tpts,1);

        for iii = 1:num_tpts
            endpt = min(num_tpts, iii + cTilt_stable_window - 1);

            cTilt_close(iii) = mean((spind_angles(iii:endpt,j))) < cTilt_thresh | ...
                (pi - mean(spind_angles(iii:endpt,j)) < cTilt_thresh);

            ikt_close(iii)   = mean((ikt_dist(iii:endpt,j))) > ikt_thresh;
        end

        biorient_time = find(cTilt_close .* ikt_close,1);
        if isempty(biorient_time)
            tt = NaN;
        else
            tt = biorient_time;
        end
        biorient_times(j) = tt;
    end


    %%%%%%%


    polepole_dist = vecnorm(p1_pos_ori-p2_pos_ori,2,2);



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%
    % first, find when the spindle starts/stops separating

    % smooths out the distance between axes
    end_smoothwindow = min([200,num_tpts]);
    smoothed_spind = smoothdata(polepole_dist(1:end_smoothwindow),1,'sgolay',50);

    normalized_smooth_spind = (smoothed_spind-min(smoothed_spind))...
        /(max(smoothed_spind)-min(smoothed_spind));

    spind_vel = diff(normalized_smooth_spind);
    smooth_spind_vel = smoothdata(spind_vel,1,'sgolay',50);

    smooth_spind_vel_norm = smooth_spind_vel./max(abs(smooth_spind_vel));
    end_sep = find((smooth_spind_vel_norm<.1)'&(1:end_smoothwindow-1)>50,1);

    spind_size = polepole_dist(end_sep);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    attach_time = zeros(tracked,1);

    for j = 1:tracked

        pole_or_center = zeros(num_tpts, 3);
        angs = zeros(num_tpts, 5);

        p1_score = zeros(num_tpts,1);
        p2_score = zeros(num_tpts,1);


        vels = squeeze(diff(chr_positions(:,j,:),1,1));




        for n = 1:num_tpts-1
            endpt = min(num_tpts-1, n+windowsize);
            vel_n = nanmean(vels(n:endpt,:),1);

            avg_pos = squeeze(nanmean(chr_positions(n:endpt,j,1:3),1));

            avg_pole1 = squeeze(nanmean(p1_pos_ori(n:endpt,1:3)));
            avg_pole2 = squeeze(nanmean(p2_pos_ori(n:endpt,1:3)));
            avg_center_pole = avg_pole1*0.5+avg_pole2*0.5;

            p1_vec = avg_pole1 - avg_pos';
            p2_vec = avg_pole2 - avg_pos';
            center_vec = avg_center_pole- avg_pos';

            vec_to_p1_norm = p1_vec/norm(p1_vec);
            vec_to_p2_norm = p2_vec/norm(p2_vec);
            vec_to_center_norm = center_vec/norm(center_vec);

            dispvec_norm = vel_n/norm(vel_n);

            dot_center = vec_to_center_norm*dispvec_norm';
            dot_p1 = vec_to_p1_norm*dispvec_norm';
            dot_p2 = vec_to_p2_norm*dispvec_norm';

            pole_or_center(n, 1) = dot_p1;
            pole_or_center(n, 2) = dot_p2;
            pole_or_center(n, 3) = dot_center;
        end



        above_dot_thresh = zeros(1,num_tpts-1);
        for iii = 1: num_tpts-1
            endpt = min(num_tpts, iii+window3-1);
            above_dot_thresh(iii) = (nanmean(pole_or_center(iii:endpt,1))>dot_thresh)|...
                (nanmean(pole_or_center(iii:endpt,2))>dot_thresh)|...
                (nanmean(pole_or_center(iii:endpt,3))>dot_thresh);

        end
        above_dot_thresh(end) = 1;
        attach_time(j) = max(neb,find(above_dot_thresh,1)+round(windowsize*.5));
    end


    start_times = attach_time;
    end_times = biorient_times;

    for j = 1:tracked
        start_j = start_times(j);
        end_j = nanmedian(biorient_times);
        times_j  = start_j:end_j;
        if length(times_j)>2
       
            p1_full = p1_pos_ori;
            p2_full = p2_pos_ori;

     

            chr_pos_j_full = squeeze(chr_positions(:,j,:));% chr_j_smooth; %squeeze(chr_positions(:,j,:));
            %
            %
            chr_pos_j = chr_pos_j_full(times_j,:);

            pole1_j = p1_pos_ori(times_j,:); %p1_smooth(times_j,:); % p1_pos_ori(times_j,:);
            pole2_j = p2_pos_ori(times_j,:); %p2_pos_ori(times_j,:);
            %
            %
            %
            init_dist_pcenter = vecnorm(chr_pos_j_full(neb,:)-polepole_cent(neb,:));
            dist_to_pcenter =  vecnorm(chr_pos_j_full(:,:)-polepole_cent(:,:),2,2);
       



            ww=30;
            traj_to_class = chr_pos_j_full-polepole_cent;
            traj_to_class = traj_to_class(neb:min(end_sep+15,num_tpts),:);
            normalized = traj_to_class./(polepole_dist(neb:min(end_sep+15,num_tpts)));

            num_tpts_cpt= length(normalized);
            trajectorylabel = zeros(1,num_tpts_cpt);
            trajectorylabel(:) = 1;
            plot_times = 1:num_tpts_cpt;


            input =table(trajectorylabel', plot_times',normalized(:,1) ,normalized(:,2), normalized(:,3),...
                'VariableNames',{'Trajectory','Time','x','y','z'});
            cpresult = cp_analysis(input,3,[],ww,1,1,SimH0Cp,SimH0);

            if isempty(cpresult)
                cpstart=NaN;
            else

                change1_padded = [1,cpresult.ChangePoint,num_tpts_cpt];
                states_long1 = zeros(1,num_tpts_cpt);
                for ccc = 1:(length(change1_padded)-1)
                    if ccc>length(cpresult.MotionType)
                        this_state = 0;
                    else
                        this_state = cpresult.MotionType(ccc);
                    end
                    start_c = change1_padded(ccc);
                    end_c = change1_padded(ccc+1);
                    states_long1(start_c:end_c)=this_state;
                end


                cpstart = states_long1(max(attach_time(j)-neb,1));
            end

            %%


            


    %%


            traj_count = traj_count+1;
            maxspind(traj_count)=max(polepole_dist);

            % arrival_times(traj_count) = arrive_time;
            dists_from_pcenter_full{traj_count}= dist_to_pcenter;
            initdists(traj_count) = init_dist_pcenter;
            chrom_positions_full{traj_count} = chr_pos_j_full;
            starts(traj_count) = start_j;
            ends(traj_count) = end_times(j);
            nebs(traj_count) = neb;
            endseps(traj_count)= end_sep;

            chrom_positions{traj_count} = chr_pos_j;
            pole1_positions{traj_count} = pole1_j;
            pole2_positions{traj_count} = pole2_j;
            dists_from_pcenter{traj_count}  = dist_from_cent(times_j,j);
            traj_celllabels(traj_count) = c;
            pole1_full{traj_count} = p1_full;
            pole2_full{traj_count} = p2_full;


        else
            bad_count = bad_count+1;

        end
    end

end

clearvars -except cellIDs filename bad_count traj_count maxspind dists_from_pcenter_full initdists  chrom_positions_full starts ends nebs endseps ...
   chrom_positions pole1_positions pole2_positions dists_from_pcenter traj_celllabels pole1_full pole2_full cpstarts bad_count%deletes all variables except X in workspace


save(filename)