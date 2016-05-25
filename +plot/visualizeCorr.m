function [ ] = visualizeCorr( corrMat, pVals, clust_idx, corrThresh, pThresh )
%VISUALIZECORR Summary of this function goes here
%   Detailed explanation goes here

    [nNeurons,~] = size(corrMat);
    
    corrDist = 1-abs(corrMat);
    
    lowD = cmdscale(corrDist);
    
    clust_vals = unique(clust_idx);
    
    %figure; hold on;
        
    % draw links between nodes
    for ii=1:nNeurons
        for jj=(ii+1):nNeurons
            if abs(corrMat(ii,jj))>corrThresh && pVals(ii,jj)<pThresh
                line([lowD(ii,1) lowD(jj,1)],[lowD(ii,2) lowD(jj,2)],'Color',[.5 .5 .5])
            end
        end
    end
    
    % draw graph nodes (with coloring based on clustering)
    for ii=1:length(clust_vals)
        curr_cluster = clust_idx==clust_vals(ii);
        plot(lowD(curr_cluster,1),lowD(curr_cluster,2),'.','MarkerSize',40,'Color',[0 .4 .8])
    end
    
    title(sprintf('links when abs(corr)>%.2f & p-val < %.3f',corrThresh,pThresh))
    
end

