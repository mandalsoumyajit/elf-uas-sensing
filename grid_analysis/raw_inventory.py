import zipfile, sys, collections
z = zipfile.ZipFile(sys.argv[1])
rows = collections.defaultdict(list)
other = []
for i in z.infolist():
    parts = i.filename.rstrip('/').split('/')
    if i.filename.endswith('.mat'):
        rows[parts[1]].append((parts[-1], '%02d:%02d' % i.date_time[3:5], i.file_size))
    elif not i.is_dir():
        other.append(i.filename)
key = lambda s: (0, int(s)) if s.isdigit() else (1, s)
for k in sorted(rows, key=key):
    v = sorted(rows[k])
    print(f'{k:>4}: ' + ', '.join(f'{n} {t} {s/1e6:.1f}MB' for n, t, s in v))
print('non-mat files:', other)
