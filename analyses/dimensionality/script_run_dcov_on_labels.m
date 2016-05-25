
% run dcov on 124 neurons + 1-d label (0 to 4)

load('../../data/sandy/jun06_2015_dataforhackathon.mat');
% returns data.sig (num_samples x num_neurons)
%       data.label (num_samples x 1) 0-4 tells behavior

%%% get N samples from each label
N = 500;

X = [];  % keeps track of neural data
Y = [];  % keeps track of labels
for ilabel = 0:4
    indices = data.label == ilabel;
    d = data.sig(indices,:)';
    
    r = randperm(size(d,2));
    d = d(:,r(1:N));
    
    X = [X d];
    Y = [Y ilabel*ones(1,N)];
    
end

num_shuffles = 1000;

r = randperm(length(Y));
Y = Y(r);

dist_X = squareform(pdist(X'));
dist_Y = squareform(pdist(Y'));
[sample_dcov, pvalue] =  dcov(dist_X, dist_Y, 'random', num_shuffles);


