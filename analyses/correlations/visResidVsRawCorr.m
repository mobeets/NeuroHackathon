
load('corr_results.mat')
resid_idx = load('idx_residual_cluster_labels.mat');
raw_idx = load('idx_raw_cluster_labels.mat');

corrThresh = 0.3;
pValThresh = 0.001;

figure; hold on;
plot.visualizeCorr(raw_corr,raw_pVal,raw_idx.idx,corrThresh,pValThresh);

figure; hold on;
plot.visualizeCorr(resid_corr,resid_pVal,resid_idx.idx,corrThresh,pValThresh);
