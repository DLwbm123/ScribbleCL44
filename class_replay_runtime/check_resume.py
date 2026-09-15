"""Exercise completed-stage restore, including replay sampling and the next task."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from setproctitle import setproctitle
import runner_core
from cl_methods import DarkExperienceReplayPlus
from run import ROOT, args_for

setproctitle('job')
torch.set_num_threads(4)
args = args_for('zs-er', True)
before, after = ROOT / 'checks/resume_before', ROOT / 'checks/resume_after'
args[args.index('--output') + 1] = str(before)
args.remove('--validation-only')
sys.argv = ['main', *args, '--max-task', '1']
runner_core.main('class')
state = torch.load(before / 's01_state.pt', map_location='cpu', weights_only=False)
buffer = DarkExperienceReplayPlus(64, 4, 0, 1)
buffer.load_state_dict(state['continual'])
assert buffer.num_seen_examples == 8 and len(buffer) == 8
buffer.current_stage = 1
np.random.seed(7)
sample = buffer.sample(torch.device('cpu'))
assert set(sample[3].tolist()) == {0} and buffer.replayed_sources[0]
args[args.index('--output') + 1] = str(after)
sys.argv = ['main', *args, '--max-task', '2', '--resume-from', str(before), '--task-epochs', '1', '2', '1']
runner_core.main('class')
result = json.loads((after / 'summary.json').read_text())
assert result['completed_stages'] == 2 and result['derpp_buffer']['num_seen_examples'] == 24
assert result['task_epochs'] == [1, 2, 1]
assert result['matrix'][0] == json.loads((before / 'summary.json').read_text())['matrix'][0]
assert result['derpp_buffer']['replayed_unique_source_counts']['0'] > 0
rows = [json.loads(line) for line in (after / 'train.jsonl').read_text().splitlines()]
assert [(r['stage'], r['epoch']) for r in rows if 'loss' in r] == [(0, 0), (1, 0), (1, 1)]
(ROOT / 'checks/resume_passed.json').write_text(json.dumps({'passed': True, 'stages': 2}) + '\n')
print('RESUME_CHECK_PASSED', flush=True)
