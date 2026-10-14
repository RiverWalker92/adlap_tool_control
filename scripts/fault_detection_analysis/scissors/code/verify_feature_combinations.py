"""Check new candidate decisions against the actual sklearn implementation."""
from pathlib import Path
import json,warnings
import numpy as np
import pandas as pd
from classify_cuts import dataset,fit,decision
from search_feature_combinations import ranking

ROOT=Path(__file__).resolve().parents[1]

def run(root=ROOT):
    root=Path(root);out=root/'outputs/combinations';data=pd.read_csv(root/'outputs/cut_features.csv')
    registry=pd.read_csv(out/'model_registry.csv').fillna('')
    table=pd.read_csv(out/'cv_metrics.csv');full=np.load(out/'full_models.npz');cv=np.load(out/'cv_decisions.npz')
    rng=np.random.default_rng(6802278);checked=0;max_coef_error=0.;max_bias_error=0.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Only one sample available.*')
        for (scenario,selection),g in table.groupby(['scenario','selection'],sort=False):
            key=f'{scenario}__{selection}';d,k=dataset(data,scenario,selection)
            # Every reported leading candidate, plus deterministic random mixed pairs.
            selected=list(ranking(g).head(3).model)
            if k==3:selected+=list(ranking(g,severity=True).head(3).model)
            selected+=list(rng.choice(g.model.to_numpy(),size=3,replace=False))
            for name in dict.fromkeys(selected):
                idx=int(registry.index[registry.model==name][0]);row=registry.iloc[idx]
                cols=[row.feature1]+([row.feature2] if row.feature2 else [])
                f=fit(d,cols,k);co,b=f[2],f[3]
                bc=full[key+'__coef'][idx,:,:len(cols)];bb=full[key+'__intercept'][idx]
                if k==2:bc=(bc[1]-bc[0])[None,:];bb=(bb[1]-bb[0])[None]
                assert np.allclose(co,bc,rtol=1e-7,atol=1e-8),(key,name,'coefficient',co,bc)
                assert np.allclose(b,bb,rtol=1e-7,atol=1e-8),(key,name,'intercept',b,bb)
                max_coef_error=max(max_coef_error,float(np.abs(co-bc).max()))
                max_bias_error=max(max_bias_error,float(np.abs(b-bb).max()))
                for held in d.run_id.unique():
                    tr=d[d.run_id!=held];te=d[d.run_id==held]
                    reference=fit(tr,cols,k);z,yp=decision(reference,te,cols,k)
                    positions=np.flatnonzero(d.run_id.to_numpy()==held)
                    actual=cv[key+'__prediction'][positions,idx]
                    assert np.array_equal(yp,actual),(key,name,held,'predictions')
                    checked+=len(te)
            print('Reference check:',scenario,selection,flush=True)
    checks=json.loads((out/'checks.json').read_text())
    checks.update(new_pair_predictions_checked_against_sklearn=checked,
                  max_reference_coefficient_error=max_coef_error,max_reference_intercept_error=max_bias_error,
                  new_candidate_reference_check_passed=True)
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    print('New-pair sklearn checks passed:',checked,'predictions.',flush=True)
    return checks

if __name__=='__main__':run()
