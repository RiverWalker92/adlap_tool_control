from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for script in ['summarize_previous_current_models.py','extract_features.py',
               'classify_cuts.py','plot_results.py','verify_results.py']:
    print('Running',script,flush=True)
    subprocess.run([sys.executable,str(root/'code'/script)],cwd=root,check=True)
print('Finished. Results:',root/'outputs_residual')
print('PNG/SVG figures:',root/'outputs_residual/figures')
