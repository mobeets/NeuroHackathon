
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
    
    
%%% compute latents

    colors = [1 0 0; 0 1 0; 0 0 1; 0 1 1];
    dim = 2;
    
    for isession = 1:length(data)
        p = u(:,dim)' * data(isession).sig';
        
        f = figure;
        
        plot(p, 'k');
        hold on;
        for ilabel = 1:4
            indices = find(data(isession).label == ilabel);

            plot(indices, p(indices), 'o', 'Color', colors(ilabel,:));
        end
        
        saveas(f, sprintf('./figs/latents/latent%d_session%d.pdf', dim, isession));
    end
    
    
    
    
    
    