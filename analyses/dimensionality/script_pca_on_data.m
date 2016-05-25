
% visualize the top latents across all time


load('../../data/sandy/exps_sandy_separated.mat');
% returns data.sig (num_samples x num_neurons)
%       data.label (num_samples x 1) 0-4 tells behavior

%%% concatenate all data

X = [];
labels = [];
for isession = 1:length(data)
    if (isession == 6)
        continue;
    end
    X = [X data(isession).sig'];
    labels = [labels data(isession).label'];
end


%%% compute PCA

    [u, sc, lat] = pca(X');
    

    p = cumsum(lat)./sum(lat);
    
    
    f = figure;
    
    plot([0 124], [0.95 0.95], 'k');
    hold on;
    plot(p);
    ylim([0 1]);
    
    saveas(f, './figs/eigenspectrum.pdf');
    
    
    
    
    
    
    