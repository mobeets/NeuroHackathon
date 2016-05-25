
% find out the number of clusters needed across neurons

load('./results/residual_data_peaks.mat');
d = residual_data.sig';


num_neurons = size(d,1);
num_folds = 6;

indices = crossvalind('Kfold', num_neurons, num_folds);


cluster_errors = [];

for inum_clusters = 1:10
    
inum_clusters
    error = 0;
    for ifold = 1:6
ifold
        Xtrain = d(indices ~= ifold,:);

        [idx, cluster_means] = kmeans(Xtrain, inum_clusters);
        
        cluster_means = cluster_means';
        Xtest = d(indices == ifold,:);
        
        for ineuron = 1:size(Xtest,1)
            squared_dists = [];
            for imean = 1:inum_clusters
                squared_dists(imean) = norm(Xtest(ineuron,:)' - cluster_means(:,imean)).^2;
            end
            error = error + min(squared_dists);
        end
        
    end
    
    cluster_errors(inum_clusters) = error;
end

f = figure;
plot(cluster_errors);
saveas(f, './figs/crossvalidated_cluster_errors_residuals.pdf');


[idx, cluster_means] = kmeans(d, 3);
save('./results/idx_residual_cluster_labels.mat', 'idx');

