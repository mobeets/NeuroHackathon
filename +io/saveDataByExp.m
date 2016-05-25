function data = saveDataByExp(data, outfile)

    trs = data.first_file_inds;
    trs = [trs size(data.sig,1)];
    ds = struct([]);
    for ii = 1:numel(trs)-1
        d = struct();
        d.sig = data.sig(trs(ii):trs(ii+1)-1,:);
        d.label = data.label(trs(ii):trs(ii+1)-1);
        ds = [ds d];
    end
    
    % save if outfile provided
    data = ds;
    if ~isempty(outfile)
        save(outfile, 'data');
    end
end
