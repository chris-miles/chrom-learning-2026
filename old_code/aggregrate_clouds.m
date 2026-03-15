clear all
close all



cell_db

ttt=1;

switch ttt
    case 1
        cellIDs=cellIDs_ctr;
        filename='ctrl_trajs_cloud';

    case 2
        cellIDs=cellIDs_cenp;
        filename='cenp_trajs_cloud';

    case 3
        cellIDs=cellIDs_dyn;
        filename='rod_trajs_cloud';
    case 4
        cellIDs=cellIDs_double;
        filename='rodgsk_trajs_cloud';
end



num_cells = length(cellIDs);

for c = 1:num_cells
    this_cell = cellIDs{c};
    celltype = strcat('../data/',this_cell,'.mat');
    load(celltype);

    % copy matrices so that everything is sychronized by NEB
    cc = centrioles(neb:size(centrioles,1),:,:);
    kk = kinetochores(neb:size(centrioles,1),:,1:tracked);

    num_tpts = length(cc);
    if num_tpts > 181
        num_tpts = 181;
        cc(182:length(cc),:,:) = [];
        kk(182:length(kk),:,:) = [];
    end

    p1_pos    = zeros(num_tpts,3);   % original ppositions of pole 1
    p2_pos    = zeros(num_tpts,3);   % original ppositions of pole 2
    chrom_center = zeros(num_tpts,3);
    chr_positions = zeros(num_tpts, tracked, 3);
    pp_cent = zeros(num_tpts,3);
    pp_vec = zeros(num_tpts,3);


    for i = 1:num_tpts
        p1_pos(i,:) = cc(i,1:3,1);
        p2_pos(i,:) = cc(i,1:3,2);

        kk_pos  = zeros(3,tracked);

        for j = 1:tracked
            k1 = kk(i,1:3,j);
            k2 = kk(i,4:6,j);
            kk_pos(:,j) = nanmean([k1;k2],1);

        end
        chr_positions(i, :,:) = kk_pos';
        chrom_center(i,:) = nanmean(kk_pos,2);

        pp_cent(i,:) = 0.5*(p1_pos(i,:)+p2_pos(i,:));

        pp_vec(i,:) = (p1_pos(i,:)-p2_pos(i,:));

    end

    polepole_dist = vecnorm(pp_vec,2,2);

    end_smoothwindow = min([200,num_tpts]);
    smoothed_spind = smoothdata(polepole_dist(1:end_smoothwindow),1,'sgolay',50);

    normalized_smooth_spind = (smoothed_spind-min(smoothed_spind))...
        /(max(smoothed_spind)-min(smoothed_spind));

    spind_vel = diff(normalized_smooth_spind);
    smooth_spind_vel = smoothdata(spind_vel,1,'sgolay',50);

    smooth_spind_vel_norm = smooth_spind_vel./max(abs(smooth_spind_vel));
    end_sep = find((smooth_spind_vel_norm<.1)'&(1:end_smoothwindow-1)>50,1);

    spind_size = polepole_dist(end_sep);




    p1_shift = p1_pos-pp_cent;
    p2_shift = p1_pos-pp_cent;
    chrom_rotated = zeros(size(chr_positions));
    qrotvals = zeros(num_tpts,4);
    u1vals = zeros(num_tpts,3);

    for i = 1:num_tpts
        chr_shift_i = squeeze(chr_positions(i,:,:))-pp_cent(i,:);
    
        uu =  p1_shift(i,:)';
        % essentially we want a rotation u1 -> u2
        u1 = uu/norm(uu);
        u2 = [1;0;0];
        u1vals(i,:) = u1;

        % if it's the first time step, just find any old rotation that works
        % now using quaternions to rotate instead
        if i==1
            qrot = qRot_u1u2(u1,u2);
            qrotvals(i,:) = qrot; % this is the quaternion encoding u1->u2 rot
        else
            % if its a LATER time step
            % first compute the rotation
            % u1new -> u1 old
            % and then ADD this rotation to the old rotation.
            old_u1 = u1vals(i-1,:);
            new_u1 = u1;

            % quaterninion u1new - > u1
            q_i_to_iplus1 = qRot_u1u2(new_u1, old_u1);

            % previous quaternion
            oldq = qrotvals(i-1,:);

            % multiply qold*qnew (using quaternion rules, order very
            % important!)
            new_q = qMul(oldq, q_i_to_iplus1);
            qrot = new_q;

            % store this quaternion
            qrotvals(i,:) = qrot;
        end

        % turn quaternion into rotation matrix
        sp_rot_mat = qGetR(qrot);


        % calculate positions of spindle poles after rotation
        p1_rotated(i,:) = sp_rot_mat * p1_shift(i,:)';
        p2_rotated(i,:) = sp_rot_mat * p2_shift(i,:)';
        chrom_rotated(i,:,:) = (sp_rot_mat * (chr_shift_i)')';
    end


  zz = squeeze(chrom_rotated(:,:,1));
   rr = sqrt(squeeze(chrom_rotated(:,:,2)).^2+squeeze(chrom_rotated(:,:,3)).^2);


    zvals{c} = zz;
    rvals{c}=rr;
    zinits{c}=zz(1,:);
    rinits{c} = rr(1,:);

    chrom_clouds{c}=chr_positions;
    chrom_centers{c}=chrom_center;
    p1_trajs{c} = p1_pos;
    p2_trajs{c} = p2_pos;
    p_centers{c} = pp_cent;
    end_seps{c} = end_sep;
    mean_spinds{c} = nanmean(p1_rotated(1:end_sep,1));
end

save(filename)