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
    
    dcorrs = zeros(num_neurons);
    
    H = eye(num_samples) - 1/num_samples * ones(num_samples);
    
    dvars = [];
    for ineuron = 1:num_neurons
ineuron
        D{ineuron} = H * squareform(pdist(d(ineuron,:)')) * H;
        dvars(ineuron) = D{ineuron}(:)' * D{ineuron}(:);
    end
    
    
    
    for ineuron = 1:num_neurons
        for jneuron = ineuron+1:num_neurons
            fprintf('neurons %d %d\n', ineuron, jneuron);
            rng(31490 + ineuron + jneuron);
            r = randperm(num_samples);
            indices = r(1:num_smaller_samples);
            tic;
            dc = sqrt(D{ineuron}(:)' * D{jneuron}(:) / dvars(ineuron) / dvars(jneuron));
            [dc, p_value] =  dcorr(D{ineuron}, D{jneuron}, 'random', 0);
            toc
            dcorrs(ineuron,jneuron) = dc;
        end
    end
    
    save('./results/dcorrs_pairwise_raw.mat', 'dcorrs');
    


%%% compute pairwise Pearson correlations
    
    corrs = zeros(num_neurons);
    
    for ineuron = 1:num_neurons
        for jneuron = ineuron+1:num_neurons
            fprintf('neurons %d %d\n', ineuron, jneuron);
            rng(31490 + ineuron + jneuron);
            r = randperm(num_samples);
            indices = r(1:num_smaller_samples);
            c = corr(d(ineuron,indices)', d(jneuron,indices)');
            corrs(ineuron,jneuron) = c;
        end
    end
    
    save('./results/corrs_pairwise_raw.mat', 'corrs');


%%% plot figure
    load('./results/dcorrs_pairwise_raw.mat');
    load('./results/corrs_pairwise_raw.mat');
    
    f = figure;
    plot(corrs, dcorrs, '.');
    
    saveas(f, './figs/dcorrs_vs_corrs_raw.pdf');




