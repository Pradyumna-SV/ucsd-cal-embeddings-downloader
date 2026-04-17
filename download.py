import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pelicanfs.core import PelicanFileSystem

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'manifest.csv')
BASE_DIR      = '/workspace/embeddings'
SUFFIXES      = ['_centers.json', '_mean.npy', '_var.npy', '_tile2vec.npy']
CHUNK_SIZE    = 1024 * 1024  # 1 MB
MAX_WORKERS   = 16

# One PelicanFileSystem per thread to avoid contention
_local = threading.local()

def get_pelfs():
    if not hasattr(_local, 'pelfs'):
        _local.pelfs = PelicanFileSystem('pelican://osg-htc.org')
    return _local.pelfs


def download_day(row):
    year  = int(row['year'])
    month = int(row['month'])
    day   = int(row['day'])
    prefix     = f'{year}_{month:02d}_{day:02d}'
    remote_dir = row['full_path']
    local_dir  = os.path.join(BASE_DIR, str(year), f'{month:02d}', f'{day:02d}')
    os.makedirs(local_dir, exist_ok=True)

    pelfs = get_pelfs()
    day_failed = False

    for suffix in SUFFIXES:
        local_path  = os.path.join(local_dir, f'{prefix}{suffix}')
        remote_path = f'{remote_dir}{prefix}{suffix}'

        if os.path.exists(local_path):
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
            print(f'  FAILED {remote_path}: {e}', flush=True)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            day_failed = True

    return (year, month, day), day_failed


print('Loading manifest...', flush=True)
manifest = pd.read_csv(MANIFEST_PATH)
queue = (
    manifest[manifest['status'] == 'OK']
    .sort_values(['year', 'month', 'day'])
    .reset_index(drop=True)
)
print(f'Days to download: {len(queue)}  |  Workers: {MAX_WORKERS}', flush=True)
os.makedirs(BASE_DIR, exist_ok=True)

success, failed = 0, 0
total = len(queue)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(download_day, row): i
        for i, row in queue.iterrows()
    }
    for future in as_completed(futures):
        i = futures[future]
        try:
            (year, month, day), day_failed = future.result()
            if day_failed:
                failed += 1
                print(f'[{i+1}/{total}] PARTIAL  {year}-{month:02d}-{day:02d}', flush=True)
            else:
                success += 1
                print(f'[{i+1}/{total}] OK       {year}-{month:02d}-{day:02d}', flush=True)
        except Exception as e:
            failed += 1
            print(f'[{i+1}/{total}] ERROR: {e}', flush=True)

print(flush=True)
print('=' * 50, flush=True)
print(f'Done.', flush=True)
print(f'  Days completed : {success}', flush=True)
print(f'  Days failed    : {failed}', flush=True)
