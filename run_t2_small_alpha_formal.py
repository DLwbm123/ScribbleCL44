"""Finish the declared small-alpha checks and conditionally run Organ T2/T3 for 60 epochs."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def choose(records, min_t1, min_t2):
    eligible = [r for r in records if r.get('exit_code') == 0 and r.get('steps') == 420
                and all(math.isfinite(r.get(k, float('nan'))) for k in ('T1', 'T2'))
                and r['T1'] >= min_t1 and r['T2'] >= min_t2]
    return max(eligible, key=lambda r: (r['T2'], r['T1'])) if eligible else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checks-root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, checks = args.root.resolve(), args.checks_root.resolve()
    with (checks / 'controller.started').open('x') as f:
        f.write(str(os.getpid()) + '\n')

    def status(state, **fields):
        record = dict(state=state, time=time.strftime('%Y-%m-%d %H:%M:%S %z'), **fields)
        tmp = checks / 'controller_status.tmp'
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        tmp.replace(checks / 'controller_status.json')
        print(json.dumps(record), flush=True)

    try:
        launch = json.loads((checks / 'launch.json').read_text())
        status('waiting_for_checks', formal_started=False)
        deadline = time.monotonic() + 3600
        while not all((checks / (c['name'] + '.exitcode')).exists() for c in launch['cases']):
            if time.monotonic() >= deadline:
                raise TimeoutError('checks did not finish within the one-hour dependency window')
            time.sleep(15)
        records = []
        for c in launch['cases']:
            case = checks / c['name']
            item = dict(name=c['name'], alpha=c['alpha'], exit_code=int(
                (checks / (c['name'] + '.exitcode')).read_text()))
            if item['exit_code'] == 0:
                protocol = json.loads((case / 'diagnostic_protocol.json').read_text())
                assert protocol['alpha'] == c['alpha'] and protocol['beta'] == .5 and protocol['clip'] == 5
                trace = [json.loads(line) for line in (case / 'run/numerical_trace.jsonl').read_text().splitlines()]
                assert len(trace) == 420
                assert all(math.isfinite(row['before_step']['gradient_norm']) and
                    all(math.isfinite(v) for v in row['before_step']['losses'].values()) for row in trace)
                completed = json.loads((case / 'BOUNDED_FINITE.json').read_text())
                bn = json.loads((case / 'bn_calibration_probe.json').read_text())
                scores = bn['results']['head']['validation']
                assert abs(scores['T1'] - completed['validation']['T1']) < 1e-6
                item.update(steps=completed['steps_completed'], **scores,
                            before_calibration=completed['validation'])
            records.append(item)
        selected = choose(records, launch['selection']['min_t1'], launch['selection']['min_t2'])
        (checks / 'selection.json').write_text(json.dumps(dict(records=records, selected=selected,
            rule=launch['selection']), indent=2, allow_nan=False) + '\n')
        if selected is None:
            status('no_candidate_passed', formal_started=False, records=records)
            return
        if args.output.exists():
            raise FileExistsError(f'formal output already exists: {args.output}')
        if shutil.disk_usage(checks).free < 20 * 1024**3:
            raise OSError('less than 20 GiB available for formal checkpoints and logs')
        probe = checks / '.formal_write_probe'
        probe.write_text('ok')
        assert probe.read_text() == 'ok'
        probe.unlink()
        gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                            '--format=csv,noheader,nounits'], text=True)
        available = [(int(memory), int(gpu)) for gpu, memory in
                     (row.split(',') for row in gpu_rows.splitlines())
                     if int(gpu) in (4, 5, 6, 7) and int(memory) >= 16000]
        if not available:
            raise RuntimeError('no authorized GPU has the required 16000 MiB free; formal not started')
        _, gpu = max(available)
        command = [sys.executable, '-u', '-c',
            'import torch; import runner_core; torch.set_num_threads(4); '
            'torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; '
            'torch.use_deterministic_algorithms(True, warn_only=True); runner_core.main("organ")',
            '--data-root', str(root / 'subset/data'), '--sparse-root', str(root / 'subset/sparse'),
            '--output', str(args.output), '--device', 'cuda:0', '--seed', '42', '--epochs-per-task', '60',
            '--max-task', '3', '--batch-size', '4', '--workers', '8', '--lr', '.03', '--method', 'zs-derpp',
            '--der-alpha', '.5', '--der-beta', '.5', '--der-buffer-size', '128', '--der-minibatch-size', '4',
            '--pce-loss-weight', '1', '--zs-global-weight', '1', '--zs-spatial-loss-weight', '.01',
            '--zs-spatial-warmup-epochs', '28', '--validate-each-epoch', '--test-evaluation',
            '--t3-first-epoch-evaluation', '--numerical-debug', '--organ-t2-supervision-strategy',
            '--organ-t2-feature-alpha', str(selected['alpha']), '--t2-from', str(root / 'run60'),
            '--annotation-id', 'organ_T13_half_train_seed42_small_alpha']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (checks / 'formal.log').open('x') as log:
            process = subprocess.Popen(command, cwd=args.source, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        launch_record = dict(pid=process.pid, gpu=gpu, command=command, selected=selected,
                             output=str(args.output), source=str(args.source))
        (checks / 'formal_launch.json').write_text(json.dumps(launch_record, indent=2) + '\n')
        status('formal_process_launched', formal_started=True, **launch_record)
        code = process.wait()
        (checks / 'formal.exitcode').write_text(str(code) + '\n')
        status('formal_exited' if code == 0 else 'formal_failed', formal_started=True, exit_code=code,
               output=str(args.output))
    except Exception as error:
        status('controller_failed', error=repr(error))
        raise


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        base = dict(exit_code=0, steps=420, T1=.65, T2=.33, alpha=.01)
        assert choose([base], .65, .33) == base
        for patch in ({'T1': .649}, {'T2': .329}, {'T2': float('nan')}, {'steps': 419}, {'exit_code': 1}):
            assert choose([{**base, **patch}], .65, .33) is None
        assert choose([base, {**base, 'alpha': .05, 'T2': .4}], .65, .33)['alpha'] == .05
        print('PASS: selection thresholds, failed/incomplete/nonfinite exclusion, and ranking')
    else:
        main()
