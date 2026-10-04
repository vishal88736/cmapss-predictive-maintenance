"""Verify every shipped training observation against the local C-MAPSS source files."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
    source = root.parent / 'data' / f'train_{subset}.txt'
    exported = root / 'public' / 'data' / f'{subset}.json'
    payload = json.loads(exported.read_text())
    assert payload['source'] == source.name and payload['split'] == 'train'
    assert len(payload['conditionNames']) == 3
    assert len(payload['sensorNames']) == 21
    actual = {}
    for line in source.read_text().splitlines():
        values = line.split()
        if not values:
            continue
        assert len(values) == 26
        unit = int(values[0])
        actual.setdefault(unit, []).append([int(values[1]), *map(float, values[2:])])
    assert len(actual) == len(payload['engines'])
    for engine in payload['engines']:
        assert engine['readings'] == actual[engine['engineId']]
        assert engine['readings'][-1][0] - engine['readings'][-1][0] == 0
    print(f'{subset}: every exported observation matches {source.name}')
