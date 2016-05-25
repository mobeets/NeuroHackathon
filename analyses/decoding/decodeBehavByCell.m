% load data

d = load('data/exps.mat');

% run on each experiment
nboots = 10;
nexps = numel(d.data);
for expNum = 1:nexps
    data = d.data(expNum);
    outdir = fullfile('fits', ['exp' num2str(expNum)])
    mkdir(outdir);
    decodeCells;
end

% compare decoding results
plot.compareDecoding(d, 'plots/decodeCells', 'mv_v_still', 'fb_v_lr');
