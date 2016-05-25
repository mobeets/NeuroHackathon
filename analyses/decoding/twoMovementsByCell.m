function [errs, allErr, obj] = decodeTwoMovementsByCell(X, Y, ix, ...
    nboots, doCv, doSingleCells)
    if nargin < 4
        nboots = 10;
    end
    if nargin < 5
        doCv = true;
    end
    if nargin < 6
        doSingleCells = true;
    end

    ncells = size(X,2);
    errs = nan(ncells, nboots);
    allErr = nan(nboots, 1);

    nn1 = sum(ix);
    inds = find(~ix);
    nn2 = numel(inds);
    if nn1 < nn2
        ix0 = ~ix;
        inds = find(ix);
        nnA = nn1; nnB = nn2;
    else
        ix0 = ix;
        nnA = nn2; nnB = nn1;
    end
 
    for jj = 1:nboots
        indsSamp = randi(nnA, [nnB 1]);
        inds1 = inds(indsSamp);
        Xc = [X(ix0,:); X(inds1,:)];
        Yc = [Y(ix0); Y(inds1)];

        if doSingleCells
            for ii = 1:ncells
                obj = fitcdiscr(Xc(:,ii), Yc, 'FillCoeffs', 'on');
                if doCv
                    cvmodel = crossval(obj,'kfold',5);
                    errs(ii,jj) = kfoldLoss(cvmodel);
                else
                    errs(ii,jj) = resubLoss(obj);
                end
            end
        end

        obj = fitcdiscr(Xc, Yc, 'FillCoeffs', 'on');
        if doCv
            cvmodel = crossval(obj,'kfold',5);
            allErr(jj) = kfoldLoss(cvmodel);
        else
            allErr(jj) = resubLoss(obj);
        end
%         confusionmat(obj.Y, resubPredict(obj))
    end

end
