% compute dim and patterns for each label, then see how dims relate


load('../../data/sandy/exps_sandy_separated.mat');
% returns data.sig (num_samples x num_neurons)
%       data.label (num_samples x 1) 0-4 tells behavior

%%% compute patterns for each label

d = [];
labels = [];
for isession = 1:length(data)
    if (isession == 6)
        continue;
    end
    d = [d data(isession).sig'];
    labels = [labels data(isession).label'];
end


vars = [];
U = [];
for ilabel = 0:4
    
    indices = labels == ilabel;
    X = d(:,indices);
    
    [u, sc, lat] = pca(X');
    
    vars(ilabel+1) = sum(lat);
    p = cumsum(lat)./sum(lat);
    opt_dim = sum(p < 0.9) + 1;
    U{ilabel+1} = u(:,1:opt_dim);
    
    fprintf('label=%d dim=%d var=%f\n', ilabel, opt_dim, vars(ilabel+1));
end


%%% compare patterns across labels
[dims, random_dims, min_dims, max_dims] = compare_patterns(U);

f = figure;
plot(dims);
hold on;

plot(min_dims, '+k');
plot(max_dims, '+k');
plot(random_dims, 'r');

saveas(f, './figs/compare_patterns.pdf');




