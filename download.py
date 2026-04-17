import os
import pandas as pd
from pelicanfs.core import PelicanFileSystem

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'manifest.csv')
BASE_DIR      = '/workspace/embeddings'
SUFFIXES   = ['_centers.json', '_mean.npy', '_var.npy', '_tile2vec.npy']
CHUNK_SIZE = 1024 * 1024  # 1 MB

print('Loading manifest...')
manifest = pd.read_csv(MANIFEST_PATH)
queue = (
    manifest[manifest['status'] == 'OK']
    .sort_values(['year', 'month', 'day'])
    .reset_index(drop=True)
)
print(f'Days to download: {len(queue)}')

pelfs = PelicanFileSystem('pelican://osg-htc.org')
os.makedirs(BASE_DIR, exist_ok=True)

success, skipped, failed = 0, 0, 0

for i, row in queue.iterrows():
    year  = int(row['year'])
    month = int(row['month'])
    day   = int(row['day'])
    prefix     = f'{year}_{month:02d}_{day:02d}'
    remote_dir = row['full_path']
    local_dir  = os.path.join(BASE_DIR, str(year), f'{month:02d}', f'{day:02d}')
    os.makedirs(local_dir, exist_ok=True)

    day_failed = False
    for suffix in SUFFIXES:
        local_path  = os.path.join(local_dir, f'{prefix}{suffix}')
        remote_path = f'{remote_dir}{prefix}{suffix}'

        if os.path.exists(local_path):
            skipped += 1
            continue

        tmp_path = local_path + '.tmp'
        try:
            with pelfs.open(remote_path, 'rb') as src, open(tmp_path, 'wb') as dst:
                while True:
                    chunk = src.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst.write(chunk)
            os.rename(tmp_path, local_path)
        except Exception as e:
            print(f'  FAILED {remote_path}: {e}')
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            day_failed = True
            failed += 1

    if day_failed:
        print(f'[{i+1}/{len(queue)}] PARTIAL  {year}-{month:02d}-{day:02d}')
    else:
        success += 1
        print(f'[{i+1}/{len(queue)}] OK       {year}-{month:02d}-{day:02d}')

print()
print('=' * 50)
print(f'Done.')
print(f'  Days completed : {success}')
print(f'  Files skipped  : {skipped}  (already existed)')
print(f'  Files failed   : {failed}')
