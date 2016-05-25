function [  ] = visualizeClust( raw_clust_idx, resid_clust_idx )
% Summary of this function goes here
%   Detailed explanation goes here
    
    nNeurons = length(raw_clust_idx);
    raw_clust_vals = unique(raw_clust_idx);
    resid_clust_vals = unique(resid_clust_idx);
    
    rng(0);
    distMat = zeros(nNeurons);
    for ii=1:nNeurons
        for jj=ii+1:nNeurons
            if raw_clust_idx(ii)==raw_clust_idx(jj)
                distMat(ii,jj) = abs(randn(1)+3);
                distMat(jj,ii) = distMat(ii,jj);
            else
                distMat(ii,jj) = abs(randn(1)+3.5);
                distMat(jj,ii) = distMat(ii,jj);
            end
        end
    end
    
    lowD = cmdscale(distMat);
    
    figure(1); hold on;
    plot(lowD(raw_clust_idx==1,1),lowD(raw_clust_idx==1,2),'b.','MarkerSize',40)
    plot(lowD(raw_clust_idx==2,1),lowD(raw_clust_idx==2,2),'r.','MarkerSize',40)
    title('Clusters from raw activity')
    
    figure(2); hold on;
    plot(lowD(resid_clust_idx==1,1),lowD(resid_clust_idx==1,2),'k.','MarkerSize',40)
    plot(lowD(resid_clust_idx==2,1),lowD(resid_clust_idx==2,2),'g.','MarkerSize',40)
    plot(lowD(resid_clust_idx==3,1),lowD(resid_clust_idx==3,2),'m.','MarkerSize',40)
    title('Clusters from residual activity')

end

