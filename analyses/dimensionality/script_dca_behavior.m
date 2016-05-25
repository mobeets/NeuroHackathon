
% apply DCA to the neural activity + behavioral data

load('../../data/sandy/data_peaks.mat');
% returns data.sig, .label


num_subsamples = size(data.sig,1);
num_DCA_dims = 5;

X = data.sig';
Y = data.label';

num_neurons = size(X,1);
num_samples = size(X,2);

rng(31490);
r = randperm(num_samples);



Xs{1} = X(:,r(1:num_subsamples));
Ds{1} = squareform(pdist(Y(r(1:num_subsamples))'));


%%% compute DCA on neural activity + behavior

%     [Us, dcovs] = dca_stoch(Xs, Ds, 'num_dca_dimensions', num_DCA_dims, 'num_stoch_batch_samples', 10, 'num_iters_across_datasets', 20);
% 
%     save('./results/Us_dcovs_behavior.mat', 'Us', 'dcovs');
    
    
%%% compute DCA on activity + shuffled behavior
disp('shuffled');
shuffled_dcovs = [];
for irun = 1:100
    tic;
    r = randperm(num_subsamples);
    Xs_shuffled{1} = Xs{1}(:,r);
    
    [Us, shuffled_dcov] = dca_stoch(Xs_shuffled, Ds, 'num_dca_dimensions', 1, 'num_stoch_batch_samples', 10, 'num_iters_across_datasets', 20);
    toc

    save(sprintf('./results/shuffled_dcovs/shuffled_dcov_run%d.mat', irun), 'shuffled_dcov');
end




