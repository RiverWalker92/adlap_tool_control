from pathlib import Path
import argparse,subprocess,sys
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path);p.add_argument('--manifest',type=Path,default=root/'runs.json');p.add_argument('--output-dir',type=Path,default=root/'outputs');a=p.parse_args()
out=a.output_dir.resolve();args=['--manifest',str(a.manifest.resolve()),'--output-dir',str(out)]
if a.data_dir:args+=['--data-dir',str(a.data_dir.resolve())]
subprocess.run([sys.executable,str(root/'code/shaft_analysis.py'),*args],cwd=root,check=True)
for script in ['plot_additional_lda.py','verify_results.py']:
 subprocess.run([sys.executable,str(root/'code'/script),'--output-dir',str(out)],cwd=root,check=True)
print('Finished. Results:',out)
print('PNG figures:',out/'figures')

# Extra waveform and motor tracking figures, using the same inputs as the analysis.
if __name__ == '__main__':
    waveform_command=[sys.executable,str(root/'code/plot_waveforms.py'),'--manifest',str(a.manifest.resolve()),'--output-dir',str(a.output_dir.resolve())]
    if a.data_dir:waveform_command+=['--data-dir',str(a.data_dir.resolve())]
    subprocess.run(waveform_command,cwd=root,check=True)
