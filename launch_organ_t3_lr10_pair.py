"""Exactly two 10-epoch T3 continuations: doubled LR, Spatial 0 versus 0.01."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

from launch_organ_metrics_formal import commands, check_completed, write_json
from run_independent_domains_formal import run_gpu_queue


def pair_jobs(root):
    base = commands(root)[0]['command']
    jobs = []
    for name, weight in (('spatial0', '0'), ('spatial001', '.01')):
        cmd = list(base)
        for flag, value in {'--lr':'.06', '--epochs-per-task':'10', '--max-task':'3',
                            '--zs-spatial-loss-weight':weight, '--zs-spatial-warmup-epochs':'-1',
                            '--output':str(root/'runs'/name)}.items():
            cmd[cmd.index(flag)+1] = value
        cmd.append('--t3-each-epoch-evaluation')
        jobs.append(dict(name=name, command=cmd, first_stage=2, stages=3, epochs=10))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); root = args.root.resolve(); jobs = pair_jobs(root)
    if args.dry_run:
        print(json.dumps(jobs, indent=2)); return
    assert root.is_relative_to(Path('/data_nas')) and (root/'checks/cpu_passed.json').is_file()
    with (root/'coordinator.started').open('x') as f:
        f.write(str(os.getpid()))
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
    for key, directory in {'TMPDIR':'tmp', 'XDG_CACHE_HOME':'cache', 'TORCH_HOME':'cache/torch',
                           'CUDA_CACHE_PATH':'cache/cuda', 'TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor',
                           'MPLCONFIGDIR':'cache/matplotlib'}.items():
        (root/directory).mkdir(parents=True, exist_ok=True); env[key] = str(root/directory)
    outcomes = {job['name']: dict(status='queued') for job in jobs}; lock = threading.Lock()
    def status(name, **fields):
        with lock:
            outcomes[name] = fields
            write_json(root/'progress.json', dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),jobs=outcomes))
    def execute(job, gpu):
        name = job['name']
        try:
            with (root/'logs'/(name+'.log')).open('x') as log:
                process = subprocess.Popen(job['command'],cwd=root/'source',env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status(name,status='running',gpu=gpu,pid=process.pid)
                code = process.wait()
            (root/'logs'/(name+'.exitcode')).write_text(str(code)+'\n')
            if code: raise RuntimeError(f'training exited {code}')
            summary = check_completed(root,job)
            curve = [json.loads(line) for line in (root/'runs'/name/'t3_retention_curve.jsonl').read_text().splitlines()]
            assert [row['epoch_one_based'] for row in curve] == list(range(1,11))
            for row in curve:
                assert all(math.isfinite(v) for v in (row['val']['t2_after'],row['test']['t2_after'],row['T3_validation'],row['T3_test']))
                assert (root/'runs'/name/row['checkpoint']).stat().st_size > 0
            status(name,status='complete',gpu=gpu,**summary)
        except Exception as error:
            status(name,status='failed',gpu=gpu,error=str(error))
    write_json(root/'launch_plan.json', dict(jobs=jobs,gpus=[4,5,6,7],max_jobs=2,min_free_mib=22000,
        initial_head_lr=.06,initial_backbone_lr=.006,LR_horizon_epochs=10,spatial_start_epoch=1,
        checkpoint_selection='T3 validation; epoch endpoint weights saved separately',class_jobs=0))
    run_gpu_queue(jobs,execute,min_free_memory_mib=22000)
    complete = all(value['status']=='complete' for value in outcomes.values())
    if complete:
        curves = {job['name']:[json.loads(line) for line in (root/'runs'/job['name']/'t3_retention_curve.jsonl').read_text().splitlines()] for job in jobs}
        write_json(root/'comparison.json',dict(curves=curves,selection='T3 validation, never test',epoch10_is_separate_endpoint=True))
        with (root/'comparison.csv').open('w',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(['epoch','T2_test_spatial0','T2_test_spatial001','T3_test_spatial0','T3_test_spatial001'])
            for left,right in zip(curves['spatial0'],curves['spatial001']):
                writer.writerow([left['epoch_one_based'],left['test']['t2_after'],right['test']['t2_after'],left['T3_test'],right['T3_test']])
    (root/'pipeline.exitcode').write_text('0\n' if complete else '1\n')
    if not complete: raise SystemExit(1)


if __name__ == '__main__': main()
