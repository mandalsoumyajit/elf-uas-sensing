% Inspect the timeseries objects in the raw grid data
root = fullfile(fileparts(mfilename('fullpath')), 'raw', '9_29 mandal raw data');
f = fullfile(root, '1', 'adcA.mat');
s = load(f);
fn = fieldnames(s);
ts = s.(fn{1});
disp(class(ts)); disp(fn{1});
disp(ts)
fprintf('Length %d, Time(1..5): %s\n', ts.Length, mat2str(ts.Time(1:5)', 8));
fprintf('Time end %.6f, median dt %.3e\n', ts.Time(end), median(diff(ts.Time)));
d = ts.Data; fprintf('Data class %s size %s min %g max %g\n', class(d), mat2str(size(d)), min(d(:)), max(d(:)));
disp(ts.TimeInfo)
disp(ts.DataInfo)
fprintf('Name: %s\n', ts.Name);
