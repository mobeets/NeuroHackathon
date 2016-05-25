%%% plot the top two dca dims for motor projections


load('./results/motor_proj_data_peaks.mat');

p = motor_proj_data.sig';

num_samples = size(p,2);

colors = [0 0 0; 0 0 1; 1 0 0];

f = figure;
for ilabel = 0:2
    indices = motor_proj_data.label == ilabel;
    plot(p(1,indices), p(2,indices), '.', 'Color', colors(ilabel+1,:), 'MarkerSize', 7);
    hold on;
end

saveas(f, './figs/top_two_dca_dims.pdf');