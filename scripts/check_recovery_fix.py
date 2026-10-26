"""Audit recovery input stability and run an optional validation-only neural pilot.

This script never opens official test trajectories or RUL files. Pilot results
are execution/learning diagnostics, not the full-run benchmark.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from grace_rul.config import DEFAULT_CONFIG, FORMAT_VERSION, CORRUPTION_PROTOCOL
from grace_rul.corruption import corrupt_trajectory
from grace_rul.data import OperatingRegimeScaler, WindowDataset, load_training_subset, split_engines
from grace_rul.normalization import encode_sensors
from grace_rul.training import device_for, evaluate, fit_pipeline


def audit_representation(frame, groups, subset):
    fit = frame[frame.unit.isin(groups['fit'])]
    val = frame[frame.unit.isin(groups['validation'])]
    linear = OperatingRegimeScaler().fit(fit, subset, 42, sensor_transform='linear')
    state = {**deepcopy(linear.state), 'sensor_transform':'asinh', 'sensor_transform_scale':1.}
    compressed = OperatingRegimeScaler(state)
    results = []
    for family in ['stuck', 'mixed', 'bias', 'noise']:
        max_linear = max_compressed = max_raw_difference = 0.
        changed_count = prefix_checks = 0
        for unit, group in val.groupby('unit', sort=True):
            values, regimes = linear.transform(group)
            encoded = values.copy(); encoded[:,3:] = encode_sensors(values[:,3:], state)
            seed = np.random.SeedSequence([DEFAULT_CONFIG['validation_seed'], 0, int(unit)])
            options = dict(family=family, severity=.2, regimes=regimes, return_metadata=True)
            old, old_labels, old_meta = corrupt_trajectory(values, np.random.default_rng(seed), profile=linear.state, **options)
            new, labels, meta = corrupt_trajectory(encoded, np.random.default_rng(seed), profile=state, **options)
            assert old_meta['domain']==meta['domain'] and old_meta['channels']==meta['channels']
            assert old_meta['onset_observed_offset']==meta['onset_observed_offset']
            np.testing.assert_array_equal(np.isnan(new),np.isnan(old))
            np.testing.assert_array_equal(new[:,:3],values[:,:3])
            np.testing.assert_array_equal(new[:,3:][~labels],encoded[:,3:][~labels])
            raw_old, raw_new = linear.inverse_sensors(old[:,3:],regimes), compressed.inverse_sensors(new[:,3:],regimes)
            np.testing.assert_allclose(raw_new,raw_old,rtol=2e-6,atol=2e-4,equal_nan=True)
            finite = np.isfinite(raw_old) & np.isfinite(raw_new)
            max_raw_difference = max(max_raw_difference,float(np.abs(raw_old-raw_new)[finite].max()))
            max_linear = max(max_linear,float(np.abs(old[:,3:][np.isfinite(old[:,3:])]).max()))
            max_compressed = max(max_compressed,float(np.abs(new[:,3:][np.isfinite(new[:,3:])]).max()))
            changed_count += int(labels.sum())
            for n in [30,min(100,len(encoded)),len(encoded)//2]:
                short, short_labels = corrupt_trajectory(encoded[:n],np.random.default_rng(seed),family,.2,
                                                         profile=state,regimes=regimes[:n])
                np.testing.assert_array_equal(short,new[:n])
                np.testing.assert_array_equal(short_labels,labels[:n])
                prefix_checks += 1
        results.append({'family':family,'severity':.2,'validation_engines':int(val.unit.nunique()),
                        'changed_entries':changed_count,'maximum_linear_input_magnitude':max_linear,
                        'maximum_asinh_input_magnitude':max_compressed,
                        'maximum_raw_difference_between_representations':max_raw_difference,
                        'prefix_checks':prefix_checks})
    return results


def neural_pilot(frame, groups, subset, output, device):
    config = {**DEFAULT_CONFIG,'device':device,'quick_run':True,'subsets':[subset],'seeds':[42],
              'tune_before_training':False,'run_matched_benchmarks':False,'run_extended_evaluation':False,
              'predictor_epochs':3,'generator_epochs':3,'joint_epochs':3,'patience':4,
              'detector_refinement_epochs':3,'detector_refinement_patience':3,
              'maximum_windows':256,'maximum_batches':8,'window_stride':5,'batch_size':16,
              'adversarial_warmup':1,'progress_interval_seconds':15.}
    fit = frame[frame.unit.isin(groups['fit'][:25])].copy()
    val = frame[frame.unit.isin(groups['validation'][:8])].copy()
    model, scaler, history, score = fit_pipeline(fit,val,subset,config,42,
                                                checkpoint_path=output/f'pilot_{subset}.pt')
    history.to_csv(output/f'pilot_{subset}_history.csv',index=False)
    model.detector_history.to_csv(output/f'pilot_{subset}_detector_history.csv',index=False)
    (output/f'pilot_{subset}_detector_audit.json').write_text(json.dumps(model.detector_audit,indent=2,allow_nan=False))
    assert np.isfinite(history.train_loss).all()
    reports = {}
    for family in ['clean','mixed','stuck']:
        data = WindowDataset(val,scaler,config,mode='validation',seed=config['validation_seed'],family=family)
        report, records = evaluate(model,data,config)
        records.to_csv(output/f'pilot_{subset}_{family}_validation.csv',index=False)
        reports[family] = report
    return {'scope':'Small same-partition validation pilot; full architecture, 25 fit / 8 validation engines; not a benchmark',
            'config':config,'fit_units':sorted(int(u) for u in fit.unit.unique()),
            'validation_units':sorted(int(u) for u in val.unit.unique()),
            'selected_repair':bool(model.repair_enabled),'selected_validation_score':score,
            'detector_output_weight_norm':float(model.detector.head.weight.detach().norm()),
            'detector_audit':model.detector_audit,
            'finite_repair_thresholds':model.finite_repair_thresholds.cpu().tolist(),
            'reports':reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subsets',nargs='+',choices=['FD002','FD004'],default=['FD002','FD004'])
    parser.add_argument('--pilot',action='store_true',help='Also train nine small-budget epochs on held-out training engines.')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    if args.pilot:
        device_for({**DEFAULT_CONFIG,'device':args.device})
    output = args.output_dir or ROOT/'artifacts'/('repair_check_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True,exist_ok=True)
    result = {'scope':'Training-file-only input audit, plus optional limited neural pilot; no official test evaluation',
              'format_version':FORMAT_VERSION,'corruption_protocol':CORRUPTION_PROTOCOL,
              'torch_version':str(torch.__version__),'subsets':{}}
    for subset in args.subsets:
        frame = load_training_subset(ROOT/'data',subset)
        groups = split_engines(frame,DEFAULT_CONFIG['split_seed'])
        item = {'training_file_sha256':hashlib.sha256((ROOT/'data'/f'train_{subset}.txt').read_bytes()).hexdigest(),
                'partitions':groups,'representation_audit':audit_representation(frame,groups,subset)}
        print(f'{subset}: physical fault equivalence and prefix checks passed',flush=True)
        if args.pilot:
            item['pilot'] = neural_pilot(frame,groups,subset,output,args.device)
        result['subsets'][subset] = item
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Saved validation-only checks: {output.resolve()}',flush=True)


if __name__ == '__main__':
    main()
