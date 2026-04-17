import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MANIFEST_PATH  = os.path.join(os.path.dirname(__file__), 'manifest.csv')
BASE_DIR       = '/workspace/embeddings'
SUFFIXES       = ['_centers.json', '_mean.npy', '_var.npy', '_tile2vec.npy']
CHUNK_SIZE     = 1 * 1024 * 1024  # 1 MB
MAX_WORKERS    = 16
REQUEST_TIMEOUT = (30, 120)  # (connect, read) seconds
OSDF_DIRECTOR  = 'https://osdf-director.osg-htc.org'


def make_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=('GET', 'HEAD'),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount('http://',  adapter)
    session.mount('https://', adapter)
    return session


session = make_session()


def download_file(remote_path, local_path):
    url      = f'{OSDF_DIRECTOR}{remote_path}'
    tmp_path = local_path + '.tmp'
    try:
        with session.get(url, stream=True, allow_redirects=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            with open(tmp_path, 'wb') as dst:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        dst.write(chunk)
        os.rename(tmp_path, local_path)
        return True, None
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False, str(e)


def download_day(row):
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
            continue

        ok, err = download_file(remote_path, local_path)
        if not ok:
            print(f'  FAILED {remote_path}: {err}', flush=True)
            day_failed = True

    return (year, month, day), day_failed


print('Loading manifest...', flush=True)
manifest = pd.read_csv(MANIFEST_PATH)
queue = (
    manifest[manifest['status'] == 'OK']
    .sort_values(['year', 'month', 'day'])
    .reset_index(drop=True)
)
total = len(queue)
print(f'Days to download: {total}  |  Workers: {MAX_WORKERS}', flush=True)
print(f'Director: {OSDF_DIRECTOR}', flush=True)
os.makedirs(BASE_DIR, exist_ok=True)

success, failed = 0, 0

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
