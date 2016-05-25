function plotDecodeGroupAvg(mu, se, grps, ttl, tag)
    plot.init;
    bar(grps, mu, 'FaceColor', 'w');
    ylim([0 0.6]);
    xts = 0:0.1:0.5;
    set(gca, 'XTick', grps);
    set(gca, 'YTick', xts, 'YTickLabel', arrayfun(@num2str, xts, 'uni', 0));
    for ii = 1:ngrps
        plot([ii ii], [mu(ii)-se(ii) mu(ii)+se(ii)], 'k-');
    end
    title([ttl '-' strrep(tag, '_', '-')]);
end
