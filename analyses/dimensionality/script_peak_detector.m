

load('../../data/sandy/hackCleanBeh.mat');
% returns data.sig, .label

X = data.sig';
Y = data.label';

num_neurons = size(X,1);
num_samples = size(X,2);



%%% plot figures of peaks

    ineuron = 5;
    [pks,loc] = findpeaks(X(ineuron,:), 'MinPeakHeight', 2.5*std(X(ineuron,:)), 'MinPeakDistance', 500);
    
    f = figure;
    plot(X(ineuron,:));
    hold on;
    plot(loc, pks, 'r*');
    saveas(f, './figs/find_peaks.pdf');
    
    

%%% find peaks for each neuron
    locs = [];
    for ineuron = 1:num_neurons
        %ineuron

        [pks,loc] = findpeaks(X(ineuron,:), 'MinPeakHeight', 2.5*std(X(ineuron,:)), 'MinPeakDistance', 500);
        locs = [locs loc];

    end

locs = unique(locs);

%%% only keep peaks
    data.sig = data.sig(locs,:);
    data.label = data.label(locs);
    
save('./results/data_peaks.mat', 'data');
