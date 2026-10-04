"""Export the local C-MAPSS training files to browser-friendly JSON, without changing values.

Run from website/: python3 scripts/export_cmapss.py
Source files remain in ../data. Training trajectories run to failure; this export
retains only observations, not inferred labels or model predictions.
"""
from collections import defaultdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'data'
DEST = ROOT / 'public' / 'data'
DEST.mkdir(parents=True, exist_ok=True)

for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
    source = SOURCE / f'train_{subset}.txt'
    if not source.exists():
        print(f'Skipping {subset}: source missing')
        continue
    engines = defaultdict(list)
    width = None
    with source.open() as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if width is None:
                width = len(parts)
            if len(parts) != width or width < 6:
                raise ValueError(f'Unexpected row width in {source}: {len(parts)}')
            engine_id = int(parts[0])
            cycle = int(parts[1])
            # Retain source decimals exactly as numeric values. No smoothing or sampling.
            engines[engine_id].append([cycle, *[float(value) for value in parts[2:]]])
    if not engines:
        raise ValueError(f'No readings found in {source}')
    for engine_id, rows in engines.items():
        cycles = [row[0] for row in rows]
        if cycles != sorted(set(cycles)):
            raise ValueError(f'Non-monotonic or duplicate cycle in {subset} unit {engine_id}')
    payload = {
        'subset': subset,
        'split': 'train',
        'source': f'train_{subset}.txt',
        'sensorNames': [f'Sensor {n}' for n in range(1, width - 4)],
        'conditionNames': ['Operational setting 1', 'Operational setting 2', 'Operational setting 3'],
        'engines': [{'engineId': key, 'readings': value} for key, value in sorted(engines.items())],
    }
    target = DEST / f'{subset}.json'
    target.write_text(json.dumps(payload, separators=(',', ':')))
    print(f'{target.relative_to(ROOT)}: {len(engines)} real training units, {sum(map(len, engines.values()))} readings')
