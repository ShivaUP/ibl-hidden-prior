"""Find a working spike-load strategy against empty ONE cache."""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.data.config import load_frozen_config, repo_root

cfg = load_frozen_config()
root = repo_root()
from one.api import ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True,
          cache_dir=str(root / cfg['data']['cache_dir']))

eid = '28741f91-c837-4147-939e-918d38d849f2'
all_ds = [str(d) for d in one.list_datasets(eid)]
cands = [d for d in all_ds if d.rsplit('/', 1)[-1] == 'spikes.times.npy'
         and d.startswith('alf/probe00/pykilosort/')]
print('candidate paths:')
for c in cands:
    print('  ', c)

full_rev = next((d for d in cands if '#' in d), None)
full_norev = next((d for d in cands if '#' not in d), None)

strategies = {
    'A collection+revision': lambda: one.load_dataset(
        eid, 'spikes.times.npy', collection='alf/probe00/pykilosort', revision='2024-05-06'),
    'B full path (rev)': lambda: one.load_dataset(eid, full_rev) if full_rev else None,
    'C full path (norev)': lambda: one.load_dataset(eid, full_norev) if full_norev else None,
    'D remote object': lambda: one.load_object(
        eid, 'spikes', collection='alf/probe00/pykilosort', query_type='remote')['times'],
    'E load_dataset remote': lambda: one.load_dataset(
        eid, 'spikes.times.npy', collection='alf/probe00/pykilosort', query_type='remote'),
}
for name, fn in strategies.items():
    try:
        out = fn()
        n = len(np.asarray(out)) if out is not None else 'None'
        print(f'{name}: OK  n={n}')
    except Exception as e:
        print(f'{name}: FAIL  {type(e).__name__}: {str(e)[:70]}')
