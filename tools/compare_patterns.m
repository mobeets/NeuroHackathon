function [dims, random_dims, min_dims, max_dims] = compare_patterns(patterns)
%
% compares the basis patterns between different conditions
%
% INPUTS:
%   patterns: (1 x M cell, where M is the number of stimuli)
%       patterns{istim} (num_variables x num_latent_dims)
%
% OUTPUTS:
%   dims: (1 x (2^M-1)), dimensionalities from rank method (see below) for each
%       combination of basis patterns (including individual patterns)
%   random_dims: (1 x (2^M-1)), dimensionalities that one would expect from
%       chance (drawn from num_variables-dimensional space)
%   min_dims: (1 x (2^M-1)), dimensionalities of best overlap
%   max_dims: (1 x (2^M-1)), dimensionalities if completely orthogonal
%
%
% uses the rank method(A,B), where A: N x M_A, B: N x M_B, 
%       N is the number of neurons, M_A is the number of latent dimensions
%       for A, and M_B is the number of latent dimensions for B
% rank([A B], thresh), where thresh is 0.5
%
% example:
%   if patterns consisted of u_A and u_B (patterns for stim A and stim B)
%   then dims(1) corresponds to u_A
%   dims(2) --> u_B
%   dims(3) --> rank([u_A u_B])
%
% author: Ben Cowley, 1/20/2015

    thresh = 0.5;  % threshold for rank method
    
    
    dims = [];
    random_dims = [];
    min_dims = [];
    max_dims = [];

    num_stim = length(patterns);
    
    num_neurons = size(patterns{1}, 1); 
    

    
    % get the number of patterns for each stimulus
    stim_dim = [];
    for istim = 1:num_stim
        stim_dim = [stim_dim; size(patterns{istim},2)];
    end
    
    for inum_combs = 1:num_stim  % number of elements in combination
        
        combs = nchoosek(1:num_stim, inum_combs);
        
        for icomb = 1:size(combs,1)
            % actual patterns
            combined_patterns = [patterns{combs(icomb,:)}];
            dims = [dims rank(combined_patterns, thresh)];
            
            % random patterns, drawn from N-dimensional space
                combined_random_patterns = [];
                for istim = combs(icomb,:)
                    R_stim = orth(randn(num_neurons, stim_dim(istim)));
                    combined_random_patterns = [combined_random_patterns R_stim];
                end
                random_dims = [random_dims rank(combined_random_patterns, thresh)];
            
            % random patterns, drawn from M-dimensional space,
            %       where M = number of patterns when aggregating all stim
%                 M = rank([patterns{1:end}], thresh);
%                 combined_random_patterns = [];
%                 for istim = combs(icomb,:)
%                     R_stim = orth(randn(M, stim_dim(istim)));
%                     combined_random_patterns = [combined_random_patterns R_stim];
%                 end
%                 random_dims = [random_dims rank(combined_random_patterns, thresh)];
            
            
            % random patterns, drawn from (kA+kB+...)-d space
%                 [Q, R] = qr(combined_patterns);
%                 Q = Q(:, 1:min(num_neurons, sum(stim_dim(combs(icomb,:)))));
%                 combined_random_patterns = [];
%                 for istim = combs(icomb,:)
%                     R = orth(randn(size(Q,2), stim_dim(istim)));
%                     R_stim = Q * R;
%                     combined_random_patterns = [combined_random_patterns R_stim];
%                 end
%                 random_dims = [random_dims rank(combined_random_patterns, thresh)];
            
            % min dims
            min_dims = [min_dims max(stim_dim(combs(icomb,:)))];
            
            % max dims
            max_dims = [max_dims min(sum(stim_dim(combs(icomb,:))), num_neurons)];
            
        end
        
    end




end