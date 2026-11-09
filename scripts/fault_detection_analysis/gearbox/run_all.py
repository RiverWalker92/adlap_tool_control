#!/usr/bin/env python3
"""Run gearbox analysis, all figures and saved-output verification."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    config = args.config.resolve()
    output = args.output.resolve()
    if not config.is_file():
        parser.error(f'Missing {config}. First run clean_project.py with the current results folder.')
    # The 300 candidates are now produced by analyse.py as cv_margin_*.
    # The old combinations/ files use a different validation protocol.
    legacy = output/'combinations'
    if legacy.is_symlink():
        parser.error(f'Refusing to remove a symbolic link: {legacy}')
    if legacy.is_dir():
        print(f'Removing obsolete combination exports: {legacy}', flush=True)
        shutil.rmtree(legacy)
    steps = [
        ['analyse.py', '--config', str(config), '--output', str(output)],
        ['plots.py', '--output', str(output)],
        ['verify.py', '--output', str(output), '--figures', str(output/'figures')],
    ]
    for script, *arguments in steps:
        print(f'\nGearbox: {script}', flush=True)
        subprocess.run([sys.executable, str(ROOT/'code'/script), *arguments],
                       check=True, cwd=ROOT)
    print(f'\nCompleted. Results: {output}\nFigures: {output / "figures"}', flush=True)


if __name__ == '__main__':
    main()
