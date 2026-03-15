clear all
close all

%load('agg_ctrl_data.mat');
load('../agg_ctr_unsmoothed.mat')

minR = 0.5;
maxR = 9.5;
smoothwindow = 1;


nTrajs = length(chrom_positions);
labels = zeros(1,nTrajs);
for j = 1:nTrajs
    labels(j) = traj_celllabels{j};
end


nCells = max(labels);




n_to_leaveout = 2;

combos = nchoosek(1:nCells, n_to_leaveout);

nCombos = length(combos);

nBasisMax = 21;

nBasisVals = 2:2:nBasisMax;
nBasisSweep = length(nBasisVals);

errs = zeros(nBasisSweep,nCombos);

for m = 1:nBasisSweep
    nBasis = nBasisVals(m);
    
    
    parfor n = 1:nCombos
        tot_indices = 1:nCells;
        this_combo = combos(n,:);
        train_set = tot_indices;
        train_set(this_combo) = [];
        
        train_counter = 1;
        test_counter = 1;
        
        chr_pos_train = {};
        chr_pos_test = {};
        p1_pos_train = {};
        p1_pos_test = {};
        p2_pos_train = {};
        p2_pos_test = {};
        
        
        for j=1:nTrajs
            this_label = labels(j);
            in_train = find(train_set== this_label);
            if isempty(in_train)
                in_train  =0;
            end
            
            if in_train
                chr_pos_train{train_counter} = chrom_positions{j};
                p1_pos_train{train_counter} = pole1_positions{j};
                p2_pos_train{train_counter} = pole2_positions{j};
                train_counter = train_counter+1;
            else
                chr_pos_test{test_counter} = chrom_positions{j};
                p1_pos_test{test_counter} = pole1_positions{j};
                p2_pos_test{test_counter} = pole2_positions{j};
                test_counter = test_counter+1;
            end
        end
        [solved_cheby1] = do_model1(chr_pos_train,p1_pos_train,p2_pos_train,minR,maxR,nBasis,smoothwindow);
        [solved_cheby2] = do_model2(chr_pos_train,p1_pos_train,p2_pos_train,minR,maxR,nBasis,smoothwindow);
        model1_err = validate_model1(chr_pos_test,p1_pos_test,p2_pos_test,minR,maxR,nBasis,smoothwindow,solved_cheby1);
        model2_err = validate_model2(chr_pos_test,p1_pos_test,p2_pos_test,minR,maxR,nBasis,smoothwindow,solved_cheby2);
        errs(m,n) = model1_err;
        
    end
    
end
