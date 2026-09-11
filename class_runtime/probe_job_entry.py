"""Installed as a neutral main entry; arguments stay out of the OS command line."""
import json
from pathlib import Path
import runpy
import sys
from setproctitle import setproctitle

setproctitle('job')
config = json.loads(Path('config.json').read_text())
source = Path(config['source'])
sys.path.insert(0, str(source))
sys.argv = ['main', *config['argv']]
runpy.run_path(str(source / 'run_class_job.py'), run_name='__main__')
