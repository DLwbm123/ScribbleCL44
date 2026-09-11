"""Selection isolation and a bounded four-task native GPU integration check."""
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from run_baseline_tuning import RECIPES,METHODS,arguments,select_recipe
from runner_core import organ_training_policy


def main():
    for method in METHODS:
        for recipe in RECIPES:
            for formal in (False,True):
                args=arguments(Path('/data_nas/check'),method,recipe,formal)
                assert ('--test-evaluation' in args)==formal
                assert args[args.index('--epochs-per-task')+1]==('40' if formal else '10')
                assert '--organ-report-schedule' not in args and '--organ-tuning' in args
                assert not any(x in args for x in ('--t2-from','--t3-from','--t4-from'))
    candidates=[dict(status='complete',recipe=r,validation_mean=.2+i*.1,test_mean=100-i)
                for i,r in enumerate(RECIPES)]
    assert select_recipe(candidates)['name']=='c3'
    for row in candidates:row['validation_mean']=.5
    assert select_recipe(candidates)['name']=='c0'
    candidates[0]['status']='failed'
    try:select_recipe(candidates)
    except ValueError:pass
    else:raise AssertionError('failed candidate permitted formal selection')
    args=SimpleNamespace(organ_t2_supervision_strategy=True,der_alpha=.5,der_beta=.5,
        grad_clip_norm=None,organ_t2_feature_alpha=.05,organ_retention_strategy=True,
        organ_tuning=True,organ_backbone_lr_scales=[.01]*3,organ_freeze_bn_from=2)
    assert 'backbone_lr_scale' not in organ_training_policy(args,'T1')
    for task in ('T2','T3','T4'):
        p=organ_training_policy(args,task)
        assert p['backbone_lr_scale']==.01 and p['freeze_backbone_bn']
    args.organ_tuning=False
    assert 'backbone_lr_scale' not in organ_training_policy(args,'T2')
    assert organ_training_policy(args,'T3')['backbone_lr_scale']==.1
    print('PASS: validation-only selection, deterministic ties, failed-screen gate, budgets, fresh runs and isolated transition controls')


def smoke():
    import torch
    import runner_core as runner
    torch.set_num_threads(4)
    root=Path(os.environ['CHECK_ROOT']);method=os.environ['CHECK_METHOD']
    # Reuse the preceding synthetic four-task fixtures, never medical images.
    fixtures=Path(os.environ['CHECK_INPUTS'])
    root.mkdir(parents=True,exist_ok=False)
    (root/'inputs').symlink_to(fixtures,target_is_directory=True)
    args=arguments(root,method,RECIPES[3],formal=False)
    args.remove('--setting-run')
    for flag,value in {'--epochs-per-task':'1','--workers':'0','--batch-size':'2',
            '--gpm-examples':'2','--gpm-max-patches-per-layer':'32',
            '--gpm-max-matrix-elements':'32768','--fisher-batches':'1'}.items():
        args[args.index(flag)+1]=value
    sys.argv=['check',*args];runner.main('organ')
    p=root/'runs'/(method+'_c3')
    s=json.loads((p/'summary.json').read_text())
    assert s['completed_stages']==4 and s['final_seen_mean'] is None
    assert all(v is None for row in s['matrix'] for v in row)
    rows=[json.loads(l) for l in (p/'train.jsonl').read_text().splitlines()]
    trained=[x for x in rows if 'loss' in x]
    assert len(trained)==4 and all(math.isfinite(x['loss']) for x in trained)
    policies=[x['task_strategy'] for x in s['stage_rows']]
    assert all(x['freeze_backbone_bn'] and x['backbone_lr_scale']==.01 for x in policies[1:])
    if method=='zs-gpm':assert all(x['gpm_gradient_ratio'] is not None for x in trained[1:])
    if method=='zs-ewc':assert len(json.loads((p/'fisher.json').read_text()))==4
    (root/'passed.json').write_text(json.dumps(dict(status='PASS',method=method,stages=4,test_evaluation=False))+'\n')
    print('PASS: four-stage tuning integration',method)


if __name__=='__main__':
    smoke() if os.environ.get('CHECK_METHOD') else main()
