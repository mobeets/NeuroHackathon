function [mu, se] = decodeCellGroupAvg(d, tag, idx)
    
    grps = unique(idx);
    ngrps = numel(grps);
    
    nexps = numel(d.data);
    aes1 = []; es1 = [];
    for expNum = 1:nexps
        if expNum == 6
            continue;
        end
        indir = fullfile('fits', ['exp' num2str(expNum)]);
        z = load(fullfile(indir, [tag '.mat']));
        es1 = [es1 mean(z.errs,2)];
        aes1 = [aes1 mean(z.allErr)];
    end

    % idx = idx1;
    mu = nan(ngrps,1);
    se = nan(ngrps,1);
    for ii = 1:ngrps
        ixgrp = grps(ii) == idx;
        e = es1(ixgrp,:); e = e(:);
        mu(ii) = mean(e);
        se(ii) = mean(e) - prctile(e, 25);
    %     se(ii) = 2*std(e);
    end
end
