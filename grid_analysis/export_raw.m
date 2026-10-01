% Export every grid timeseries to plain arrays (scipy-readable v7 .mat) plus a metadata CSV.
here = fileparts(mfilename('fullpath'));
root = fullfile(here, 'raw', '9_29 mandal raw data');
out  = fullfile(here, 'raw_export');
if ~exist(out, 'dir'), mkdir(out); end
fid = fopen(fullfile(out, 'metadata.csv'), 'w');
fprintf(fid, 'cell,file,channel,var,ts_name,header_created,n,t_start,t_end,dt_median,dt_max_dev,frac_repeat,data_min,data_max\n');
for cell = 1:25
    files = dir(fullfile(root, num2str(cell), '*.mat'));
    for k = 1:numel(files)
        f = fullfile(files(k).folder, files(k).name);
        % creation time from the MAT header text
        h = fileread(f); h = h(1:116);
        tok = regexp(h, 'Created on: (.*\d{4})', 'tokens', 'once'); if isempty(tok), tok = {''}; end
        s = load(f); fn = fieldnames(s); ts = s.(fn{1});
        x = ts.Data(:); t = ts.Time(:);
        d = diff(t);
        ch = regexp(files(k).name, 'adc([A-D])', 'tokens', 'once');
        rep = mean(diff(double(x)) == 0);
        fprintf(fid, '%d,%s,%s,%s,%s,%s,%d,%.6f,%.6f,%.9g,%.3g,%.5f,%d,%d\n', cell, files(k).name, ch{1}, fn{1}, ts.Name, ...
            strtrim(tok{1}), numel(x), t(1), t(end), median(d), max(abs(d - median(d))), rep, min(x), max(x));
        t0 = t(1); dt = median(d); %#ok<NASGU>
        save(fullfile(out, sprintf('c%02d_%s.mat', cell, ch{1})), 'x', 't0', 'dt', '-v7');
    end
    fprintf('cell %d done\n', cell);
end
fclose(fid);
