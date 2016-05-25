

%%% project data onto DCA dimensions (both full + peaked)

load('../../data/sandy/hackCleanBeh.mat');
% returns data.sig, .label

load('./results/Us_dcovs_behavior.mat');
% take top two DCs

motor_proj_data.sig = (Us{1}(:,1:2)' * data.sig')';
motor_proj_data.label = data.label;

save('./results/motor_proj_data_full.mat', 'motor_proj_data');

load('../../data/sandy/data_peaks.mat');
motor_proj_data.sig = (Us{1}(:,1:2)' * data.sig')';
motor_proj_data.label = data.label;

save('./results/motor_proj_data_peaks.mat', 'motor_proj_data');




%%% get residuals (no motor dimensions, but still in high-d space)

load('../../data/sandy/hackCleanBeh.mat');
% returns data.sig, .label

load('./results/Us_dcovs_behavior.mat');
% take top two DCs

% get orthogonal space
[Q, R] = qr(Us{1}(:,1:2));
Uorth = Q(:,3:end);


residual_data.sig = (Uorth * Uorth' * data.sig')';
residual_data.label = data.label;

save('./results/motor_proj_data_full.mat', 'motor_proj_data');

load('../../data/sandy/data_peaks.mat');
residual_data.sig = (Uorth * Uorth' * data.sig')';
residual_data.label = data.label;

save('./results/motor_proj_data_peaks.mat', 'motor_proj_data');




%%% get residuals (no motor dimensions, but still in high-d space)

load('../../data/sandy/hackCleanBeh.mat');
% returns data.sig, .label

load('./results/Us_dcovs_behavior.mat');
% take top two DCs

% get orthogonal space
[Q, R] = qr(Us{1}(:,1:2));
Uorth = Q(:,3:end);


residual_data.sig = (Uorth * Uorth' * data.sig')';
residual_data.label = data.label;

save('./results/residual_data_full.mat', 'residual_data');

load('../../data/sandy/data_peaks.mat');
residual_data.sig = (Uorth * Uorth' * data.sig')';
residual_data.label = data.label;

save('./results/residual_data_peaks.mat', 'residual_data');




%%% get residuals (no motor dimensions, but still in high-d space)

load('../../data/sandy/hackCleanBeh.mat');
% returns data.sig, .label

load('./results/Us_dcovs_behavior.mat');
% take top two DCs

% get orthogonal space
[Q, R] = qr(Us{1}(:,1:2));
Uorth = Q(:,3:end);


residual_proj_data.sig = (Uorth * Uorth' * data.sig')';
residual_proj_data.label = data.label;

save('./results/residual_proj_data_full.mat', 'residual_proj_data');

load('../../data/sandy/data_peaks.mat');
residual_proj_data.sig = (Uorth * Uorth' * data.sig')';
residual_proj_data.label = data.label;

save('./results/residual_proj_data_peaks.mat', 'residual_proj_data');



