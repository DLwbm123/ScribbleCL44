"""Run one matched domain variant: T2 for 20 epochs, then one standardized T3 epoch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base-root', type=Path, required=True)
    p.add_argument('--domain', choices=['ucl', 'bidmc'], required=True)
    p.add_argument('--gpu', type=int, choices=[4, 5, 6, 7], required=True)
    args = p.parse_args()
    folder = args.root/args.domain
    reference = json.loads((args.base_root/'small_alpha_checks_20260908/formal_launch.json').read_text())
    command = list(reference['command'])
    for flag, value in [('--data-root', folder/'data'), ('--sparse-root', folder/'sparse'),
                        ('--output', folder/'t2'), ('--max-task', 2),
                        ('--annotation-id', 'organ_domain_swap_20260909_'+args.domain)]:
        command[command.index(flag)+1] = str(value)
    command.remove('--t3-first-epoch-evaluation')
    command += ['--organ-t2-epochs', '20']
    if (folder/'execution.json').exists():
        raise FileExistsError('variant already launched')
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
               '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < 16000:
        raise RuntimeError(f'GPU {args.gpu} has only {free} MiB free')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
    record = dict(domain=args.domain, gpu=args.gpu, phases=[])
    for phase in ('t2', 't3'):
        if phase == 't3':
            source_manifest = json.loads((folder/'t2/manifest.json').read_text())
            assert source_manifest['organ_t2_executed_epochs'] == 20
            rows = [json.loads(l) for l in (folder/'t2/train.jsonl').read_text().splitlines()]
            assert sum('epoch_seconds' in row and row['stage'] == 1 for row in rows) == 20
            for flag in ('--t2-from', '--organ-t2-epochs'):
                index = command.index(flag); del command[index:index+2]
            command[command.index('--output')+1] = str(folder/'t3')
            command[command.index('--max-task')+1] = '3'
            command += ['--t3-one-epoch-from', str(folder/'t2'), '--t3-first-epoch-evaluation']
        with (folder/(phase+'.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=args.root/'source', env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
        entry = dict(phase=phase, pid=process.pid, command=list(command), started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
        record['phases'].append(entry)
        (folder/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
        code = process.wait()
        entry.update(exit_code=code, ended_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
        (folder/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
        (folder/(phase+'.exitcode')).write_text(str(code)+'\n')
        if code:
            raise SystemExit(code)
    retention = json.loads((folder/'t3/t3_epoch1_t2_retention.json').read_text())
    (folder/'complete.json').write_text(json.dumps(dict(status='complete', domain=args.domain,
        scope='20 T2 epochs and 1 T3 epoch; validation selection', retention=retention), indent=2)+'\n')


if __name__ == '__main__':
    main()
