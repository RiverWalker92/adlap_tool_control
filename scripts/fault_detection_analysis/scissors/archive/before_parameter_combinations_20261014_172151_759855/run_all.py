"""Rebuild the single residual analysis in outputs/ from the unchanged logs."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
print('Scissors analysis:',root,flush=True)
print('Current residuals only; one observation per cut; whole runs held out.',flush=True)
for script in ['extract_features.py','classify_cuts.py','plot_results.py','verify_results.py']:
    print('Running',script,flush=True)
    subprocess.run([sys.executable,str(root/'code'/script)],cwd=root,check=True)
print('Finished. Results:',root/'outputs')
print('Figures:',root/'outputs/figures')
