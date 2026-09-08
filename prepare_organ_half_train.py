"""Create patient-stratified half-training views; original datasets remain intact.

HDF5 virtual datasets select training slices without copying images. Other
datasets link to the originals. Scribble arrays use the same selected indices.
"""
import argparse
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np


TASKS = {'T1': 'UtahI.h5', 'T2': 'UCL.h5', 'T3': 'Lits.h5'}


def half_indices(ends, seed):
    ends = np.asarray(ends, dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(np.r_[-1, ends]) < 2):
        raise ValueError('Every retained patient must have at least two training slices')
    starts = np.r_[0, ends[:-1] + 1]
    counts = ends - starts + 1
    target = (int(counts.sum()) + 1) // 2
    quotas = counts // 2
    rng = np.random.default_rng(seed)
    odd = np.flatnonzero(counts % 2)
    quotas[rng.permutation(odd)[:target - int(quotas.sum())]] += 1
    chosen = np.concatenate([np.sort(rng.choice(np.arange(start, end + 1), size=int(q), replace=False))
                             for start, end, q in zip(starts, ends, quotas)])
    return chosen, quotas


def write_view(source, annotation, output_h5, output_npz, seed):
    source = source.resolve()
    with h5py.File(source, 'r') as original:
        indices, quotas = half_indices(original['patient_info_train'][:], seed)
        n = int(original['train_images'].shape[2])
        if int(original['patient_info_train'][-1]) != n - 1:
            raise ValueError('Training patient boundaries do not cover the images')
        sparse = np.load(annotation, allow_pickle=False)['annotations']
        if sparse.shape != (n, *original['train_images'].shape[:2]):
            raise ValueError('Scribbles and images do not align')
        with h5py.File(output_h5, 'x', libver='latest') as view:
            for key in original:
                if key in ('train_images', 'train_labels'):
                    dataset = original[key]
                    layout = h5py.VirtualLayout(shape=(*dataset.shape[:2], len(indices)), dtype=dataset.dtype)
                    virtual_source = h5py.VirtualSource(str(source), key, shape=dataset.shape)
                    for target, index in enumerate(indices):
                        layout[:, :, target] = virtual_source[:, :, int(index)]
                    view.create_virtual_dataset(key, layout)
                elif key == 'patient_info_train':
                    view.create_dataset(key, data=np.cumsum(quotas) - 1)
                else:
                    view[key] = h5py.ExternalLink(str(source), key)
        selected = sparse[indices]
        with output_npz.open('xb') as stream:
            np.savez_compressed(stream, annotations=selected)
        stats = {'original_train_slices': n, 'selected_train_slices': len(indices),
                 'train_patients': len(quotas), 'all_train_patients_retained': bool((quotas > 0).all()),
                 'original_foreground_slices': int((sparse == 1).any(axis=(1, 2)).sum()),
                 'selected_foreground_slices': int((selected == 1).any(axis=(1, 2)).sum()),
                 'selected_foreground_pixels': int((selected == 1).sum()),
                 'selected_background_pixels': int((selected == 0).sum())}
    return indices, stats


def self_test():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root/'source.h5'
        images = np.broadcast_to(np.arange(11), (2, 2, 11)).copy()
        labels = images % 2
        with h5py.File(source, 'x') as h:
            h['train_images'] = images
            h['train_labels'] = labels
            h['patient_info_train'] = [3, 6, 10]
            h['val_images'] = images[:, :, :2]
            h['test_images'] = images[:, :, -2:]
        np.savez_compressed(root/'original.npz', annotations=labels.transpose(2, 0, 1))
        indices, stats = write_view(source, root/'original.npz', root/'view.h5', root/'selected.npz', 42)
        assert len(indices) == 6 and len(np.unique(indices)) == 6 and stats['all_train_patients_retained']
        assert np.array_equal(indices, half_indices([3, 6, 10], 42)[0])
        with h5py.File(root/'view.h5') as h:
            assert np.array_equal(h['train_images'][:], images[:, :, indices])
            assert np.array_equal(h['train_labels'][:].transpose(2, 0, 1), np.load(root/'selected.npz')['annotations'])
            assert isinstance(h.get('val_images', getlink=True), h5py.ExternalLink)
            assert np.array_equal(h['test_images'][:], images[:, :, -2:])
    print('Patient retention, repeatable indices, image/scribble alignment and unchanged evaluation links: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--sparse-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--halve', nargs='+', choices=['T1', 'T2', 'T3'], default=['T1', 'T3'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(x is None for x in (args.data_root, args.sparse_root, args.output)):
        parser.error('data-root, sparse-root and output are required')
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output/'data/Task_incre'
    annotations = args.output/'sparse/organ'
    data.mkdir(parents=True)
    annotations.mkdir(parents=True)
    private, public = {}, {'seed': args.seed, 'halved_tasks': args.halve,
        'selection': 'fixed random sampling within each training patient, half rounded up globally',
        'validation_test_unchanged': True, 'tasks': {}}
    for task, filename in TASKS.items():
        source = args.data_root/'Task_incre'/filename
        annotation = args.sparse_root/'organ'/f'{task}_v2_s2_seed42.npz'
        if task in args.halve:
            indices, stats = write_view(source, annotation, data/filename, annotations/annotation.name, args.seed)
            private[task] = {'source': str(source.resolve()), 'train_indices': indices.tolist()}
        else:
            (data/filename).symlink_to(source.resolve())
            (annotations/annotation.name).symlink_to(annotation.resolve())
            with h5py.File(source) as h:
                n = int(h['train_images'].shape[2])
                stats = {'original_train_slices': n, 'selected_train_slices': n,
                         'train_patients': len(h['patient_info_train']), 'all_train_patients_retained': True}
        public['tasks'][task] = stats
    (args.output/'private_indices.json').write_text(json.dumps(private, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(public, indent=2)+'\n')
    print(json.dumps(public, indent=2))


if __name__ == '__main__':
    main()
