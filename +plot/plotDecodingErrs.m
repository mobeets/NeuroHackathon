function plotDecodingErrs(errs, allErr)
    [ncells, nboots] = size(errs);

    allErrMu = mean(allErr);
    allErrStd = std(allErr)/sqrt(nboots);
    errsMu = mean(errs,2);
    errsStd = std(errs, [], 2)/sqrt(nboots);

    plot.init;
    errorbar(1:ncells, errsMu, errsStd, '.');
    plot(errsMu, '.');
    xl = xlim;
    plot(xl, [0.5 0.5], 'k--');
    plot(xl, [allErrMu allErrMu], 'r--');
    plot(xl, [allErrMu-allErrStd allErrMu-allErrStd], 'r-');
    plot(xl, [allErrMu+allErrStd allErrMu+allErrStd], 'r-');

    xlabel('cell #');
    ylabel('fraction misclassified');
    xlim([0 ncells+1]);
    ylim([0 1]);    

end
