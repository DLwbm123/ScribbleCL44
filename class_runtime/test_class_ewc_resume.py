"""CPU check: resume the selected model/Fisher while changing only EWC strength."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
import runner_core as r


def main():
    with tempfile.TemporaryDirectory() as directory, patch.object(r, '_native_backbone', torch.nn.Identity):
        source = Path(directory)
        model = r.ClassModel()
        old = r.OnlineEWC(1, .1)
        old.consolidate(model, {k: torch.full_like(v, .01) for k, v in model.importance_named_parameters()})
        torch.save({'stage': 0, 'method': 'zs-ewc', 'model': model.state_dict(),
                    'continual': old.state_dict()}, source / 's01_state.pt')
        (source / 'summary.json').write_text(json.dumps({
            'stage_rows': [{'stage': 0}], 'source_train_sizes': {'0': 1500},
            'parameter_counts': [sum(p.numel() for p in model.parameters())]}))
        resumed = r.ClassModel()
        stronger = r.OnlineEWC(1000, .1)
        row, count, _ = r.restore_class_ewc_t1(resumed, stronger, source, torch.device('cpu'))
        assert row['stage'] == 0 and count == 1500 and stronger.lambda_ == 1000
        assert stronger.penalty(resumed).item() == 0
        resumed.activate_stage(1)
        with torch.no_grad():
            resumed.background.conv.weight.add_(.1)
        penalty = stronger.penalty(resumed)
        penalty.backward()
        assert penalty.item() > 0 and resumed.background.conv.weight.grad.abs().sum() > 0
        assert set(stronger.anchor) == set(old.anchor)
        assert not any(name.startswith('blocks.1.') for name in stronger.anchor)
    print('Selected T1 model/Fisher restoration and new-coefficient gradient: PASS')


if __name__ == '__main__':
    main()
