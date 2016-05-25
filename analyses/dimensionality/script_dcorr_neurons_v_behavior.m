% compute distance correlation for pairs of neurons



% compare the patterns for halfsets 
%  does PCA return reliable dimensions?  How many dimensions are reliable?

load('../../data/sandy/data_peaks.mat');
% returns data.sig (num_samples x num_neurons)
%       data.label (num_samples x 1) 0-4 tells behavior



%%% get data
d = [];
labels = [];
for isession = 1:length(data)
    d = [d data(isession).sig'];
    labels = [labels data(isession).label'];
end

num_samples = size(d,2);
num_neurons = size(d,1);


num_smaller_samples = num_samples;

%%% compute pairwise distance correlations
    
    dcorrs = [];
    
    for ineuron = 1:num_neurons
            fprintf('neurons %d %d\n', ineuron, jneuron);
            rng(31490 + ineuron);
            r = randperm(num_samples);
            indices = r(1:num_smaller_samples);
            tic;
            dist_X = squareform(pdist(d(ineuron,indices)'));
            dist_Y = squareform(pdist(labels(indices)'));
            [dc, p_value] =  dcorr(dist_X, dist_Y, 'random', 0);
            toc
            dcorrs(ineuron) = dc;
    end
    
    save('./results/dcorrs_pairwise_behavior.mat', 'dcorrs');
    






