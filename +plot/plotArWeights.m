
data = d.data(1);
ncells = size(data.sig,2);
N = 15*5;

nc = ncells;
nc = 10;

cs = nan(nc,1);
ws = nan(nc,N);
for ii = 1:nc
    Y = data.sig(:,ii);
    mdl = ar(Y, N);
    w = getpvec(mdl);
    ws(ii,:) = w;
    plot.init; plot(w, 'LineWidth', 3);
    xlabel('# lags');
    ylabel('weight');
    
    zstat = getpvec(mdl)./sqrt(diag(getcov(mdl)));
    ps = 2*(1 - normcdf(zstat));
    cs(ii) = sum(ps <= 0.05);
end

plot.init; hist(cs,20)

%%

mu = mean(ws);
se = std(ws)/sqrt(nc);
plot.init;
plot(mu, 'LineWidth', 3);
plot(mu+se, 'Color', [0.2 0.2 0.8]);
plot(mu-se, 'Color', [0.2 0.2 0.8]);
xlabel('# lags');
ylabel('weight');
