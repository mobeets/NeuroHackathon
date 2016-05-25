function [ processed_labels ] = continuous_behavior( raw_labels )
%CONTINUOUS_BEHAVIOR Summary of this function goes here
%   Detailed explanation goes here
    
    freq = 15;
    
    forwardBack_idx = raw_labels==1 | raw_labels==3;
    leftRight_idx = raw_labels==2 | raw_labels==4;
    
    raw_labels(forwardBack_idx) = 1;
    raw_labels(leftRight_idx) = 2;
    
    move_vs_rest = raw_labels~=0;
    smooth_move_rest = conv(double(move_vs_rest),ones(1,freq),'same');
    raw_labels(smooth_move_rest==0) = nan;
    
    start_move = ~isnan(raw_labels(2:end)) & isnan(raw_labels(1:end-1));
    end_move = ~isnan(raw_labels(1:end-1)) & isnan(raw_labels(2:end));
    
    start_move = find(start_move);
    end_move = find(end_move);
    
    if length(start_move)~=length(end_move)
        error('number of start movement and end movement dont match')
    end
    
    processed_labels = zeros(size(raw_labels));
    for ii=1:length(start_move)
        curr_labels = raw_labels(start_move(ii):end_move(ii));
        num1 = sum(curr_labels==1);
        num2 = sum(curr_labels==2);
        if num2>num1
            processed_labels(start_move(ii):end_move(ii)) = 2;
        else
            processed_labels(start_move(ii):end_move(ii)) = 1;
        end
    end
    
    subplot(2,1,1);
    plot(raw_labels(1:2000));
    processed_labels = medfilt1(processed_labels,2*freq-1);
    xlim([0 2000])
    
    subplot(2,1,2);
    plot(processed_labels(1:2000));
    xlim([0 2000])
    
    
end

