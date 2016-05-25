function [ pCorr_mat ] = pCorr( X )
%UNTITLED Summary of this function goes here
%   X - nSamples x nDims

    [nSamples,nDims] = size(X);

    pCorr_mat = zeros(nDims,nDims);
    
    covX = cov(X);
    precX = inv(covX);
    for ii=1:nDims
        for jj=1:nDims
            pCorr_mat(ii,jj) = -precX(ii,jj)/sqrt(precX(ii,ii)*precX(jj,jj));
        end
    end
    
end

