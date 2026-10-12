from pathlib import Path
import argparse,subprocess,sys
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=root/'config.json');p.add_argument('--output',type=Path,default=root/'outputs');a=p.parse_args()
out=a.output.resolve();figures=out/'figures'
for script,args in [('analyse.py',['--config',str(a.config.resolve()),'--output',str(out)]),('export_report.py',['--output',str(out),'--figures',str(figures)]),('check_outputs.py',['--output',str(out),'--figures',str(figures)])]:
 subprocess.run([sys.executable,str(root/'code'/script),*args],cwd=root,check=True)
print('Finished. Results:',out)
print('PNG figures:',figures)
