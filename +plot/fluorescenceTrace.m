
d = load('data/exps.mat');

for ll = 1:numel(d.data)
    data = d.data(ll);
    sig = zscore(data.sig);
    plot.init;
    for c = 0:4
        ix = data.label == c;
        sc = sig;
        sc(~ix,:) = nan;
        mkr = '.';
%         if c > 0
%             mkr = 'o';
%         end
        xs = 1:size(sc,1);
        plot(xs/15, mean(sc,2), mkr, 'Color', 'k');
    end
    title(['exp ' num2str(ll)]);
    xlabel('time (secs)');
    ylabel('\Delta F/F');
    saveas(gcf, fullfile('plots', 'exps', [num2str(ll) '-fluor.png']));
    
end
