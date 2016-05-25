function [mu, se] = decodeGroup(data, tag, idx)

    Y = data.label;
    if strcmp(tag, 'mv_v_still')
        Y(Y > 0) = 1;
    else
        Y = Y(Y > 0); Y(Y == 2) = 0;
    end
    ix = (Y == 1);

    grps = unique(idx);
    ngrps = numel(grps);
    counts = grpstats(idx, idx, @numel);
    mincnt = min(counts);

    nboots = 5;
    errs = nan(ngrps, nboots);
    for ii = 1:ngrps
        ixgrp = (idx == grps(ii));
        X = data.sig(:,ixgrp);
        for jj = 1:nboots
            inds = randi(sum(ixgrp), mincnt, 1);
            Xc = X(:,inds);
            [~, errs(ii,jj), ~] = twoMovementsByCell(Xc, Y, ix, ...
                1, true, false);
        end
    end
    mu = mean(errs,2);
    se = std(errs, [], 2)/sqrt(nboots);

end
