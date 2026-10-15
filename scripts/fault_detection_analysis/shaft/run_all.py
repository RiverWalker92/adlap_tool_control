"""Run the current shaft analysis, PNG plots and numerical checks."""
from pathlib import Path
import argparse, os, subprocess, sys
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
common=root.parent/'common'
sys.path[:0]=[str(root/'code'),str(common)]
os.environ['PYTHONPATH']=os.pathsep.join([str(common),str(root/'code'),os.environ.get('PYTHONPATH','')])
os.environ['MPLBACKEND']='Agg'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
p=argparse.ArgumentParser(description=__doc__)
mode=p.add_mutually_exclusive_group()
mode.add_argument('--plots-only',action='store_true',help='Regenerate PNG figures and compact CSV tables from existing results.')
mode.add_argument('--verify-only',action='store_true',help='Check the saved analysis results.')
mode.add_argument('--combinations-only',action='store_true',help='Reuse features and rerun the parameter-combination analysis (gearbox/scissors).')
a=p.parse_args()
out=root/'outputs';out.mkdir(exist_ok=True)
def command(script,*args):
    subprocess.run([sys.executable,str(root/'code'/script),*map(str,args)],cwd=root,check=True)

if a.combinations_only:p.error('--combinations-only applies to gearbox and scissors.')
if not a.plots_only and not a.verify_only:
    command('shaft_analysis.py','--manifest',root/'runs.json','--output-dir',out,'--skip-figures')
if not a.verify_only:
    import plots
    plots.run(out)
    command('waveforms.py','--manifest',root/'runs.json','--output-dir',out)
if not a.plots_only:command('verify.py','--output-dir',out)
print('Results:',out)
