
z = load('data/hackCleanBeh.mat');
data = z.data;

%%

z = load('data/cleaned_behavior.mat');
data = saveDataByExp(z.data, fullfile('data', 'cleaned_exps.mat'));

%% data by exp

trs = data.first_file_inds;
trs = [trs size(data.sig,1)];
ds = struct([]);
for ii = 1:numel(trs)-1
    d = struct();
    d.sig = data.sig(trs(ii):trs(ii+1)-1,:);
    d.label = data.label(trs(ii):trs(ii+1)-1);
    ds = [ds d];
end

%% xcorr

data = d.data(1);
neur = data.sig(:,1);
lbl = double(data.label == 0);
[r, lags] = xcorr(neur, lbl);
plot(lags, r)

%% temperature of each neuron for each behavior

data = d.data(1);
lbl = data.label;

sts = nan(size(data.sig,2),numel(unique(lbl)));
for col = 1:size(data.sig,2)
    sps = data.sig(:,col);
    ls = data.label(:);
    st = grpstats(sps, ls, @mean);
    sts(col,:) = st./max(st);
end
figure;
imagesc(sts);

%% # of trials per behavior

for ll = 1:numel(d.data)
    data = d.data(ll);
    
    figure; set(gcf, 'color', 'w');
    lbs = sort(unique(data.label));
    mus = grpstats(data.label, data.label, @numel);
    bar(lbs(2:end), mus(2:end));
    title(['exp ' num2str(ll)]);
    ylabel('# timepoints');
%     saveas(gcf, fullfile('plots', 'counts', [num2str(ll) '.png']));
end

%%

figure;
ls = sort(unique(lbl));
for ii = 1:numel(ls)
    subplot(2,3,ii); hold on;
    ix = lbl == ls(ii);
    plot(sps(ix));
end

%%

lbl = data.label;
del = diff(lbl);
tinds = find(del ~= 0);
tinds = [1; tinds];
sig = cell(numel(tinds)-1,1);
% sigs = nan(numel(tinds)-1,500,2500);
lbs = nan(numel(tinds)-1,1);
for ii = 1:numel(tinds)-1
    t1 = tinds(ii)+1;
    t2 = tinds(ii+1);
    if t1 == t2
        continue;
    end
    lb = data.label(t1:t2);    
    assert(numel(unique(lb)) == 1);
    lbs(ii) = lb(1);
    sg = data.sig(t1:t2,:);
    sig{ii} = sg;
end

%%

for c = 0%:4
    ix = lbs == c;
    ss = sig(ix);    
    ns = cellfun(@(x) size(x,1), ss);
    sgs = nan(numel(ss), max(ns), size(ss{1},2));
        
    for jj = 1:numel(ss)
        s = ss{jj};
        sgs(jj, 1:size(s,1), 1:size(s,2)) = s;
    end    
    
    figure; hold on;
    mu = nanmean(sgs,1);
    plot(mu
    
%     [c prctile(ns, 95) max(ns)]
end
