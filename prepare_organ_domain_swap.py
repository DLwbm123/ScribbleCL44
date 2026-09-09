"""Build isolated UCL/BIDMC T2 views with the recovered, identical Organ scribble recipe."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import h5py
import numpy as np
from organ_area_scribbles import generate, generate_wsl_style_from_baseline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root', type=Path, required=True)
    p.add_argument('--bidmc', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, path in [('ucl', args.base_root/'subset/data/Task_incre/UCL.h5'), ('bidmc', args.bidmc)]:
        with h5py.File(path, 'r') as f:
            dense = np.asarray(f['train_labels']).astype(np.int16).transpose(2, 0, 1)
            assert set(np.unique(dense)) == {0, 1}
            splits = {s: dict(slices=int(f[s+'_images'].shape[2]), patients=len(f['patient_info_'+s]))
                      for s in ('train', 'val', 'test')}
            assert np.isfinite(f['train_images'][:, :, 0]).all()
        baseline, baseline_stats = generate(dense, 0, 42)
        labels, stats = generate_wsl_style_from_baseline(dense, baseline, 0, 42, 3., 20.)
        assert (dense[labels == 1] == 1).all() and (dense[labels == 0] == 0).all()
        match = None
        if name == 'ucl':
            existing = np.load(args.base_root/'subset/sparse/organ/T2_v2_s2_seed42.npz', allow_pickle=False)['annotations']
            match = bool(np.array_equal(existing, labels))
            if not match:
                raise ValueError('Recovered recipe differs from the existing UCL annotations')
        folder = args.output/name
        (folder/'data/Task_incre').mkdir(parents=True)
        (folder/'sparse/organ').mkdir(parents=True)
        for task, filename in [('T1', 'UtahI.h5'), ('T2', 'UCL.h5'), ('T3', 'Lits.h5')]:
            source = path if task == 'T2' else args.base_root/'subset/data/Task_incre'/filename
            (folder/'data/Task_incre'/filename).symlink_to(source.resolve())
            annotation = folder/'sparse/organ'/f'{task}_v2_s2_seed42.npz'
            if task == 'T2':
                np.savez_compressed(annotation, annotations=labels)
            else:
                annotation.symlink_to((args.base_root/'subset/sparse/organ'/annotation.name).resolve())
        record = dict(dataset=name, source=str(path.resolve()), splits=splits, seed=42,
            baseline_stats=asdict(baseline_stats), stats=asdict(stats), foreground_multiplier=3,
            background_multiplier=20, foreground_coverage=stats.foreground_pixels/stats.dense_foreground_pixels,
            foreground_fraction_of_known=stats.foreground_pixels/(stats.foreground_pixels+stats.background_pixels),
            ucl_existing_recipe_exact_match=match, labels_used='training only; integer conversion identical to loader',
            comparison='same 20 epochs; different slice and update counts')
        (folder/'sparse/organ/T2_v2_s2_seed42.json').write_text(json.dumps(record, indent=2)+'\n')
        report[name] = record
    (args.output/'annotation_audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
