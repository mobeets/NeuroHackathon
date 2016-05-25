
ttl = 'residual';
% ttl = 'raw';
tag = 'mv_v_still';
% tag = 'fb_v_lr';

d = load('data/data_peaks.mat'); data = d.data;
idx = load(['data/idx_' ttl '_cluster_labels.mat']); idx = idx.idx;
grps = unique(idx);

% decode using entire group
% [mu, se] = decode.decodeGroup(data, tag, idx);
% decode using single cells in group, then take mean
[mu, se] = decodeCellGroupAvg(d, tag, idx);

plot.plotDecodeGroupAvg(mu, se, grps, ttl, tag);
