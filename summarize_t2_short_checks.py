"""Export scalar-only outcomes from the bounded Organ transition checks."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for launch in sorted(args.root.glob('launch*.json')):
        for case in json.loads(launch.read_text())['cases']:
            directory = args.root/case['name']
            protocol = json.loads((directory/'diagnostic_protocol.json').read_text())
            rows = [json.loads(line) for line in (directory/'run/numerical_trace.jsonl').read_text().splitlines()]
            values = [row['before_step'] for row in rows]
            assert all(math.isfinite(v['gradient_norm']) and
                       all(math.isfinite(loss) for loss in v['losses'].values()) for v in values)
            result = {'name': case['name'], 'gpu': case['gpu'], 'planned_steps': case['steps'],
                      'successful_steps': len(rows), 'protocol': protocol,
                      'status': 'incomplete', 'max_gradient_norm': max(v['gradient_norm'] for v in values),
                      'max_replay_feature_loss': max(v['losses']['derpp_feature'] for v in values),
                      'first_pce': values[0]['losses']['pce'], 'last_pce': values[-1]['losses']['pce'],
                      'spatial_nonzero_steps': sum(v['losses']['spatial'] != 0 for v in values)}
            clip = protocol.get('clip')
            result['clipped_steps'] = None if clip is None else sum(v['gradient_norm'] > clip for v in values)
            success, failure = directory/'BOUNDED_FINITE.json', directory/'run/FIRST_NONFINITE.json'
            assert not (success.exists() and failure.exists())
            if success.exists():
                result['status'] = 'passed'
                result['completion'] = json.loads(success.read_text())
                assert len(rows) == case['steps'] == result['completion']['steps_completed']
                assert all(math.isfinite(score) for score in result['completion']['validation'].values())
            elif failure.exists():
                data = json.loads(failure.read_text())
                result.update(status='numerical_failure', failure_step=data['iteration_one_based'],
                              failure_branch=data['message'])
            cases.append(result)
    report = {'scope': 'bounded T2 transition diagnosis; validation only; formal training not started',
              'source_T1_validation_dice': .7593914365346025,
              'all_checks_finished': all(case['status'] != 'incomplete' for case in cases),
              'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'cases': len(cases), 'finished': report['all_checks_finished']}))


if __name__ == '__main__':
    main()
