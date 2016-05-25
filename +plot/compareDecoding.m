function compareDecoding(d, basedir, tag1, tag2)
%     tag1 = 'mv_v_still';
%     tag2 = 'fb_v_lr';
%     tag1 = 'l_v_r';
%     tag2 = 'f_v_b';

    nexps = numel(d.data);
    aes1 = []; aes2 = [];
    es1 = []; es2 = [];
    for expNum = 1:nexps
        if expNum == 6
            continue;
        end
        indir = fullfile(basedir, ['exp' num2str(expNum)]);
        z = load(fullfile(indir, [tag1 '.mat']));
        es1 = [es1 mean(z.errs,2)];
        aes1 = [aes1 mean(z.allErr)];

        z = load(fullfile(indir, [tag2 '.mat']));
        es2 = [es2 mean(z.errs,2)];
        aes2 = [aes2 mean(z.allErr)];
    end

    plot.init;
    es = cat(3, es1, es2);
    for ii = 1:size(es,2)
        subplot(3,4,ii); hold on;
        e = squeeze(es(:,ii,:));
        plot(e(:,1), e(:,2), '.');
        xlim([0 1]); ylim(xlim);
    end
    xlim([0 1]); ylim(xlim);
    xlabel(strrep(tag1, '_', '-')); ylabel(strrep(tag2, '_', '-'));

    plot.init;
    aes = [aes1; aes2];
    plot(aes(1,:), aes(2,:), 'o');
    xlim([0 1]); ylim(xlim);
    xlabel(strrep(tag1, '_', '-')); ylabel(strrep(tag2, '_', '-'));

end
