
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

num_neurons = size(d,1);

%%% perform pca for every 10 neurons
    num_runs = 10;
    
    nums_samples = [100 500 1000 5000 10000 20000];
    dims = [];
    for inum = 1:length(nums_samples)
        
        fprintf('num_neurons %d\n', nums_samples(inum));
        for irun = 1:num_runs
            r = randperm(num_samples);
            [u, sc, lat] = pca(d(:,r(1:nums_samples(inum)))');
            p = cumsum(lat)./sum(lat);
            dims(irun, inum) = sum(p < 0.9) + 1;
        end
    end
        
        
    f = figure;

    m = mean(dims,1);
    s = std(dims,0,1)/sqrt(num_runs);

    errorbar(nums_samples, m, s);
        
    
    saveas(f, './figs/sweep_num_samples.pdf');



    

