from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for script in ['extract_features.py','classify_cuts.py','plot_results.py','verify_results.py']:
 print('Running',script,flush=True)
 subprocess.run([sys.executable,str(root/'code'/script)],cwd=root,check=True)
print('Finished. Results:',root/'outputs')
print('PNG figures:',root/'outputs/figures')
