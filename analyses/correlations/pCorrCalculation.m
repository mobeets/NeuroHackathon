
load('residual_data_peaks.mat')
residData = residual_data.sig;
clear residual_data;
load('data_peaks.mat')
rawData = data.sig;
clear data;

[resid_corr,resid_pVal] = corr(residData);
[raw_corr,raw_pVal] = corr(rawData);

save('pCorr_results.mat','resid_corr','raw_corr','resid_pVal','raw_pVal');

figure;
imagesc(raw_corr); colorbar;
figure;
imagesc(resid_corr); colorbar;
