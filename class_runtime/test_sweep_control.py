"""python test_sweep_control.py; no GPU or training imports needed."""
from pathlib import Path
import tempfile
from run_class_job import formal_training_paused


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    formal = ['--output', str(root/'jobs'/'ind_T1_formal')]
    assert not formal_training_paused(formal)
    (root/'SWEEP_ONLY.json').write_text('{}')
    assert formal_training_paused(formal)
    assert formal_training_paused(['--output', str(root/'jobs'/'cl_spatial_formal')])
    assert not formal_training_paused(['--output', str(root/'jobs'/'cl_spatial_0.001')])
    assert not formal_training_paused(['--output', str(root/'jobs'/'ind_T2_lr0.03')])
    assert not formal_training_paused([])
print('PASS: control blocks formal jobs and permits sweeps')
