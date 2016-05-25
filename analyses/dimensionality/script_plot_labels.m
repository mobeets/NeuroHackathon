% make a quick plot to show how labels are spastic, then smoothed

load('../../data/sandy/jun06_2015_dataforhackathon.mat');

labels_unsmoothed = data.label(10000:15000);

colors = [0, 0, 0; ...
        166,206,227; ...
        178,223,138; ...
        31,120,180; ...
        51,160,44]/255;


f = figure;
for ilabel = 0:4
    indices = labels_unsmoothed == ilabel;

    plot(find(indices), labels_unsmoothed(indices), 'o', 'Color', colors(ilabel+1,:));
    hold on;
end
ylim([-1 5]);
saveas(f, './figs/unsmoothed_labels.pdf');

load('../../data/sandy/hackCleanBeh.mat');
labels_smoothed = data.label(10000:15000);

colors = [0, 0, 0; ...
        166,206,227; ...
        178,223,138]/255;
    
f = figure;
unique(labels_smoothed)
for ilabel = 0:2
    indices = labels_smoothed == ilabel;
    plot(find(indices), labels_smoothed(indices), 'o', 'Color', colors(ilabel+1,:));
    hold on;
end

ylim([-1 3]);

saveas(f, './figs/smoothed_labels.pdf');