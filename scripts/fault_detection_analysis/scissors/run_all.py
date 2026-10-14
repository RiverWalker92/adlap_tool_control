"""Base analysis and exhaustive two-parameter comparisons in one project."""
from pathlib import Path
import argparse,subprocess,sys

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--combinations-only',action='store_true',help='Reuse existing extracted features and baseline results.')
parser.add_argument('--skip-combinations',action='store_true',help='Run only the original 121-model analysis.')
args=parser.parse_args()
if args.combinations_only and args.skip_combinations:parser.error('Choose only one mode.')
if not args.combinations_only:
    for script in ['extract_features.py','classify_cuts.py','plot_results.py','verify_results.py']:
        print('Running',script,flush=True)
        subprocess.run([sys.executable,str(root/'code'/script)],cwd=root,check=True)
if not args.skip_combinations:
    for script,extra in [('search_feature_combinations.py',['--nested']),
                         ('verify_feature_combinations.py',[]),('export_combination_models.py',[]),('plot_feature_combinations.py',[])]:
        print('Running',script,flush=True)
        subprocess.run([sys.executable,str(root/'code'/script),*extra],cwd=root,check=True)
print('Finished. Original results:',root/'outputs')
print('Parameter combinations:',root/'outputs/combinations')
