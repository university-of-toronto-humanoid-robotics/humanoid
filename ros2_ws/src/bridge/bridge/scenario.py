"""Scenario files (sim/scenarios/*.yaml): a scene, a spawn pose, a policy and a motion script for one sim test.

Kept free of MuJoCo so launch files can read a scenario too.
"""
from pathlib import Path

import yaml


def repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'sim' / 'scenarios').is_dir():
            return parent
    raise FileNotFoundError('No sim/scenarios above this file: build with --symlink-install')


def in_repo(path):
    path = Path(path)
    return str(path if path.is_absolute() else repo_root() / path)


def load_scenario(name):
    """A scenario by name (sim/scenarios/<name>.yaml) or by path."""
    path = Path(name) if name.endswith('.yaml') else repo_root() / 'sim' / 'scenarios' / f'{name}.yaml'
    if not path.is_file():
        known = ', '.join(p.stem for p in sorted((repo_root() / 'sim' / 'scenarios').glob('*.yaml')))
        raise FileNotFoundError(f'No scenario at "{path}". Known: {known}')
    with open(path) as f:
        return yaml.safe_load(f)
