function [d, p_value] =  dcorr(dist_X, dist_Y, varargin)
% [dcorr, pvalue] =  dcorr(dist_X, dist_Y, 'option', option_value)
%
% Computes the distance correlation between two datasets, where random vectors X
% and Y can be different dimensionalities.  User inputs the distance
% matrices for X and Y, and function outputs the distance correlation and
% a p-value where the null hypothesis is that X and Y are independent. 
% User has the option of a random permutation test or a shifted permutation
% test (where the latter is for time-series data).
% Uses parfor.
%
% INPUT:
%   dist_X: N x N, where N is the number of samples
%   dist_Y: N x N, where N is the number of samples
%
%   Optional arguments:
%   dcorr(dist_X, dist_Y, 'random', num_shuffles) selects the
%       random permutation test (default).  Choose the num_shuffles (1x1)
%       for the test (default = 1000).
%   dcorr(dist_X, dist_Y, 'shifted', time_window) selects the
%       shifted permutation test.  The time_window (1 x T) is a vector
%       of time point indices to compute the sample null distribution by
%       shifting the time points (circular shift).  
%       E.g. time_window = 100:200 will shift the time points such that
%       the 100th time point is repositioned to the 1st time point, and
%       so on.
%      
%
% OUTPUT:
%   dcorr: scalar between 0 and 1, the distance correlation
%   pvalue: scalar between 0 and 1, probability to accept the null 
%       hypothesis that X and Y are independent.  Computed by random
%       permutation test (all time points are permuted randomly) or
%       shifted permutation test (i.e., randomly permuting Y to break any
%       dependence but keeping time structure intact).
%
% Author:
%   bcowley, 7/21/2015


    if (nargin == 2)
        permute_type = 'random';
        num_shuffles = 1000;
    elseif (strcmp(varargin{1}, 'random'))
        permute_type = 'random';
        num_shuffles = varargin{2};
    elseif (strcmp(varargin{1}, 'shifted'))
        permute_type = 'shifted';
        time_window = varargin{2};
    
        if (any(~ismember(time_window, 1:size(dist_X,2))))
            error('time window must contain indices in 1:size(dist_X,2)');
        end
    end


    % compute the dcorr of the original data
    d = dcorr_plain(dist_X,dist_Y);
    
    % compute distribution of shuffled data (permutation test)
    if (strcmp(permute_type, 'random'))
        % RANDOM PERMUTATION TEST
        
        if (num_shuffles == 0)
            p_value = inf;
            return;
        end
        
        dcorr_shuffled = zeros(1, num_shuffles);
        
        parfor ishuffle = 1:num_shuffles
            dist_Y_shuffled = permute_matrix(dist_Y);
            dcorr_shuffled(ishuffle) = dcorr_plain(dist_X, dist_Y_shuffled);
        end
        
        % compute p_value
        p_value = sum(dcorr_shuffled >= d) / num_shuffles;
        
        if (p_value == 0)
            p_value = 1 / num_shuffles;
        end
        
    elseif (strcmp(permute_type, 'shifted'))
        % SHIFTED PERMUTATION TEST
        
        dcorr_shuffled = zeros(1, length(time_window));
    
        parfor itime = 1:length(time_window)
            dist_Y_shuffled = shift_matrix(dist_Y, time_window(itime));
            dcorr_shuffled(itime) = dcorr_plain(dist_X, dist_Y_shuffled);
        end
        
        p_value = sum(dcorr_shuffled >= d) / length(time_window);
        
        if (p_value == 0)
            p_value = 1 / length(time_window);
        end
    end
    

    if (p_value == 0)           % to give it a lower bound
        p_value = 1/length(time_window);
    end


end

function d = dcorr_plain(dist_X,dist_Y)
% helper function

    N = size(dist_X,2);
    if (N ~= size(dist_Y,2)) % Y has different number of trials
        error('Error: dist_X and dist_Y have different number of samples');
    end

    % using the formula to compute the sample distance covariance
    
    % compute the euclidean distance matrices
    a = dist_X;
    b = dist_Y;


    % compute the row and column means
    a_kbar = mean(a,2);
    a_lbar = mean(a,1);
    a_bar = mean(a_lbar);
    b_kbar = mean(b,2);
    b_lbar = mean(b,1);
    b_bar = mean(b_lbar);
    
    % compute the adjusted matrices, A and B
    A = a - repmat(a_kbar, 1, N) - repmat(a_lbar, N, 1) + a_bar*ones(size(a));
    B = b - repmat(b_kbar, 1, N) - repmat(b_lbar, N, 1) + b_bar*ones(size(b));
    

    % compute the sample distance covariance
    dCov = sqrt(1/N^2 * A(:)' * B(:));
    
    % compute the sample distance variances
    dvarX = sqrt(1/N^2 * A(:)' * A(:));
    dvarY = sqrt(1/N^2 * B(:)' * B(:));
    
    % compute the sample distance correlation
    d = dCov / sqrt(dvarX * dvarY);
    
    if (isnan(d))  % no variance in the distance means a constant, 
        d = 0;     % so independent
    end


end


function permuted_Z = permute_matrix(Z)
% function randomly permutes the samples of the given symmetric matrix
% such that elements are still arranged in symmetric order

    N = size(Z,1);
    r = randperm(N);
    
    permuted_Z = nan(size(Z));
    for irow = 1:N
        permuted_Z(irow,:) = Z(r(irow), r);
    end

end


function shifted_Z = shift_matrix(Z, itime)
% function randomly shifts the samples of the given symmetric matrix
% to account for random processes being related (but not dependent)
% which can lead to false positives for non-iid processes
    
    N = size(Z,1);
    
    r = [itime:N 1:(itime-1)];
    
    shifted_Z = nan(size(Z));
    for irow = 1:N
        shifted_Z(irow,:) = Z(r(irow), r);
    end

end