"""Export reported full-data candidates in the existing predict_cuts JSON format."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from classify_cuts import SCENARIOS

ROOT=Path(__file__).resolve().parents[1]

def run(root=ROOT,requested=None,scenario='healthy_all',selection='all'):
    root=Path(root);out=root/'outputs/combinations'
    registry=pd.read_csv(out/'model_registry.csv').fillna('')
    tops=pd.read_csv(out/'top_models.csv').drop_duplicates(['scenario','selection','model'])
    if requested:
        tops=pd.DataFrame([dict(scenario=scenario,selection=selection,model=name) for name in requested])
    arrays=np.load(out/'full_models.npz');exports=[]
    data=pd.read_csv(root/'outputs/cut_features.csv')
    for row in tops.itertuples():
        idx=int(registry.index[registry.model==row.model][0]);features=registry.iloc[idx]
        cols=[features.feature1]+([features.feature2] if features.feature2 else [])
        key=f'{row.scenario}__{row.selection}'
        class_co=arrays[key+'__coef'][idx,:,:len(cols)]
        class_b=arrays[key+'__intercept'][idx]
        k=SCENARIOS[row.scenario][1]
        if k==2:co=(class_co[1]-class_co[0])[None,:];b=(class_b[1]-class_b[0])[None]
        else:co=class_co;b=class_b
        levels=SCENARIOS[row.scenario][0];cuts={'all':[1,2,3],'first':[1],'later':[2,3]}[row.selection]
        d=data[data.iteration.isin(levels)&data.cut.isin(cuts)]
        X=d[cols].to_numpy();reference=(X@class_co.T+class_b).argmax(1)
        score=X@co.T+b;yp=(score[:,0]>0).astype(int) if k==2 else score.argmax(1)
        assert np.array_equal(reference,yp),(row.scenario,row.selection,row.model)
        exports.append(dict(scenario=row.scenario,selection=row.selection,model=row.model,parameters=cols,
                            coefficients=co.tolist(),intercept=b.tolist(),class_labels=SCENARIOS[row.scenario][2],
                            positive_rule='score>0 (standard)' if k==2 else 'argmax of three discriminant scores',
                            fit_type='descriptive full-data fit; not independent validation',n_train=len(d)))
    filename='fitted_models.json' if not requested else 'requested_fitted_models.json'
    (out/filename).write_text(json.dumps(exports,indent=2))
    print('Exported',len(exports),'full-data candidates:',out/filename,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',nargs='*');p.add_argument('--scenario',default='healthy_all');p.add_argument('--selection',choices=['all','first','later'],default='all')
    args=p.parse_args();run(requested=args.model,scenario=args.scenario,selection=args.selection)
