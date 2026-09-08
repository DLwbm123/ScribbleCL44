"""Small contract check for independent Class heads and sparse label remapping."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import h5py
import numpy as np
import torch
import runner_core as r


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for task in r.TASKS['class']:
            count = len(task.classes)
            sparse = np.full((1, 256, 256), -100, dtype=np.int16)
            sparse[0, 0, :count + 1] = [0, *task.classes]
            np.savez(root/'labels.npz', annotations=sparse)
            with h5py.File(root/'data.h5', 'w') as h:
                h['train_images'] = np.zeros((256, 256, 1), dtype=np.float32)
            d = r.H5Slices(root/'data.h5', 'train', root/'labels.npz', sparse_label_shift=task.label_shift)
            _, label = d[0]
            assert set(label.unique().tolist()) == {-100, 0, *range(1, count + 1)}
            r.native_target(label[None], count + 1)
            d.close()
            with patch.object(r, '_native_backbone', torch.nn.Identity):
                model = r.ClassModel((count,))
            assert model.output_channels(0) == count + 1 and len(model.blocks) == 1
            assert model.forward_logits(torch.zeros(2, 64, 8, 8)).shape == (2, count + 1, 8, 8)
    from zs_gco_compat import cut_grid_graph
    unary = np.array([[[0., 10.], [10., 0.]], [[0., 10.], [10., 0.]]])
    pairwise = np.array([[0., 1.], [1., 0.]])
    labels = cut_grid_graph(unary, pairwise, np.ones((1, 2)), np.ones((2, 1)))
    assert np.array_equal(labels, [[0, 1], [0, 1]])
    print('GCO known-optimum grid check: PASS')
    print('T1/T2/T3 local labels, background/ignore retention and isolated output-head checks: PASS')


if __name__ == '__main__':
    main()
