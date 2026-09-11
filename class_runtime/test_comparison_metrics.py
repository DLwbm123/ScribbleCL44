"""Checks for dense label mapping, Dice aggregation and replay source accounting."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import h5py
import numpy as np
import torch
import runner_core as r
from cl_methods import DarkExperienceReplayPlus
from run_formal_supplement import metrics, JOBS


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'task.h5'
        labels = np.zeros((256, 256, 1), dtype=np.int16)
        labels[0, :2, 0] = [1, 2]
        with h5py.File(path, 'w') as h:
            h['train_images'] = np.zeros_like(labels, dtype=np.float32)
            h['train_labels'] = labels
        ds = r.H5Slices(path, 'train', dense_supervision=True, label_shift=3, return_index=True)
        _, target, index = ds[0]
        assert index == 0 and set(target.unique().tolist()) == {0, 4, 5}
        ds.close()
    prediction = torch.tensor([[[0, 0]], [[0, 1]]])
    target = torch.tensor([[[0, 1]], [[0, 1]]])
    masks = torch.nn.functional.one_hot(prediction, 2).permute(0, 3, 1, 2).float()
    model = torch.nn.Identity().train()
    with patch.object(r, 'zs_forward', return_value={'pred_masks': masks}):
        score = r.evaluate(model, [(torch.zeros(2, 1, 1, 2), target)], np.array([0, 1]), 'cpu', None, (1,))
    assert abs(score['benchmark_mean']-.5) < 1e-5
    assert abs(score['background_inclusive_mean']-2/3) < 1e-5 and model.training
    buffer = DarkExperienceReplayPlus(buffer_size=3, minibatch_size=3)
    values = torch.arange(3.).view(3, 1, 1, 1)
    buffer.add_data(values, values, values[:, 0].long(), torch.zeros(3).long(), 2, torch.tensor([3, 3, 7]))
    buffer.sample('cpu')
    assert buffer.replayed_sources == {}
    buffer.current_stage = 1
    buffer.sample('cpu'); buffer.sample('cpu')
    assert buffer.summary()['replayed_unique_source_counts'] == {'0': 2}
    with patch('cl_methods.reservoir_index', return_value=1):
        buffer.add_data(values[:1], values[:1], values[:1, 0].long(), torch.ones(1).long(), 3, torch.tensor([11]))
    assert buffer.source_indices == [3, 11, 7]
    buffer.current_stage = 2
    buffer.sample('cpu')
    assert buffer.summary()['replayed_unique_source_counts'] == {'0': 2, '1': 1}
    summary = dict(method='zs-derpp-mib', parameter_counts=[100, 110, 120],
                   derpp_buffer=buffer.summary(), source_train_sizes={'0': 10, '1': 10},
                   whole_class_dice={'background_inclusive_mean': .75},
                   stage_rows=[{'evaluated': {f'T{i+1}': {'background_inclusive_mean': v}
                               for i, v in enumerate(row)}} for row in [[.8], [.7, .6], [.6, .6, .9]]])
    result = metrics(summary, [.6, .9])
    assert abs(result['A_Dice']-.7) < 1e-10 and abs(result['BWTR']+.125) < 1e-10
    assert result['RMA'] == 1 and abs(result['MPE']-.1) < 1e-10 and abs(result['DRR']-.15) < 1e-10
    assert len(JOBS) == 8 and all(job['spatial'] == 0 for job in JOBS)
    print('Dense mapping, inclusive Dice, unchanged foreground reporting, replay IDs and benchmark formulas: PASS')


if __name__ == '__main__':
    main()
