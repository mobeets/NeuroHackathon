
% compare the patterns for halfsets 
%  does PCA return reliable dimensions?  How many dimensions are reliable?

load('../../data/sandy/exps_sandy_separated.mat');
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

r = randperm(num_samples);
data_half1 = d(:,r(1:floor(num_samples/2)));
data_half2 = d(:,r(ceil(num_samples/2):end));

%%% perform pca on both datasets
    [u1, sc1, lat1] = pca(data_half1');
    [u2, sc2, lat2] = pca(data_half2');

%%% perform projected variance
    Sigma1 = cov(data_half1');
    Sigma2 = cov(data_half2');
    
    p_half1_onto_half1 = cumsum(diag(u1' * Sigma1 * u1)) / sum(lat1);
    p_half1_onto_half2 = cumsum(diag(u2' * Sigma1 * u2)) / sum(lat1);

    p_half2_onto_half1 = cumsum(diag(u1' * Sigma2 * u1)) / sum(lat2);
    p_half2_onto_half2 = cumsum(diag(u2' * Sigma2 * u2)) / sum(lat2);
    
    
    f = figure;
    subplot(1,2,1);
    plot(p_half1_onto_half1, 'b');
    hold on;
    plot(p_half1_onto_half2, 'r');
    
    subplot(1,2,2);
    plot(p_half2_onto_half1, 'b');
    hold on;
    plot(p_half2_onto_half2, 'r');
    
    saveas(f, './figs/proj_var_halfsets.pdf');
    

