
%% moving forward/back vs. moving left/right

X = data.sig;
Y = data.label;
Y(Y == 3) = 1;
Y(Y == 4) = 2;
ix = data.label ~= 0;
X = X(ix,:);
Y = Y(ix,:);
ix = Y == 1;
[errs, allErr] = twoMovementsByCell(X, Y, ix, nboots);

fnm = fullfile(outdir, 'fb_v_lr.mat');
save(fnm, 'errs', 'allErr');

plot.plotDecodingErrs(errs, allErr);
saveas(gcf, fullfile(outdir, 'fb_v_lr.png'));

%% moving left vs. moving right

X = data.sig;
Y = data.label;
ix = data.label == 1 | data.label == 3;
X = X(ix,:);
Y = Y(ix,:);
ix = Y == 1;
if sum(~ix) > 0 && sum(ix) > 0
    [errs, allErr] = twoMovementsByCell(X, Y, ix, nboots);

    fnm = fullfile(outdir, 'l_v_r.mat');
    save(fnm, 'errs', 'allErr');

    plot.plotDecodingErrs(errs, allErr);
    saveas(gcf, fullfile(outdir, 'l_v_r.png'));
end

%% moving forward vs. back

X = data.sig;
Y = data.label;
ix = data.label == 2 | data.label == 4;
X = X(ix,:);
Y = Y(ix,:);
ix = Y == 2;

if sum(ix) > 0 && sum(~ix) > 0
    [errs, allErr] = twoMovementsByCell(X, Y, ix, nboots);
    
    fnm = fullfile(outdir, 'f_v_b.mat');
    save(fnm, 'errs', 'allErr');

    plot.plotDecodingErrs(errs, allErr);
    saveas(gcf, fullfile(outdir, 'f_v_b.png'));
end

%% moving vs. not moving

X = data.sig;
Y = double(data.label == 0);
ix = Y == 0;
[errs, allErr] = twoMovementsByCell(X, Y, ix, nboots);

fnm = fullfile(outdir, 'mv_v_still.mat');
save(fnm, 'errs', 'allErr');

plot.plotDecodingErrs(errs, allErr);
saveas(gcf, fullfile(outdir, 'mv_v_still.png'));

