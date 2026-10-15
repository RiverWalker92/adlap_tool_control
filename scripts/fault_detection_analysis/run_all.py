"""Run the three instrument analyses from one entry point."""
from pathlib import Path
import argparse,subprocess,sys,os
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',choices=['all','gearbox','shaft','scissors'],default='all')
mode=p.add_mutually_exclusive_group()
mode.add_argument('--plots-only',action='store_true')
mode.add_argument('--verify-only',action='store_true')
mode.add_argument('--combinations-only',action='store_true')
a=p.parse_args()
projects=['gearbox','shaft','scissors'] if a.project=='all' else [a.project]
if a.combinations_only and 'shaft' in projects:p.error('Choose --project gearbox or scissors with --combinations-only.')
flag=['--plots-only'] if a.plots_only else ['--verify-only'] if a.verify_only else ['--combinations-only'] if a.combinations_only else []
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['MPLBACKEND']='Agg'
for project in projects:
    print('Running',project,flush=True)
    subprocess.run([sys.executable,str(root/project/'run_all.py'),*flag],cwd=root/project,env=env,check=True)
