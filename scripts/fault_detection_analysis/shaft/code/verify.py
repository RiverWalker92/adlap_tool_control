"""Independent analytical and saved-result checks; no additional model fitting."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from shaft_features import current_parameters,definitions
from shaft_analysis import counts


def verify(out=Path(__file__).resolve().parents[1]/'outputs'):
    out=Path(out);checks=[]
    # Exact constant waveform and known Fourier amplitudes at native timestamps.
    n=1000;t=np.arange(n)/n;phase=2*np.pi*t
    constant,_=current_parameters(t,np.full(n,4.),0.,1.)
    assert np.isclose(constant['mean'],4.) and np.isclose(constant['absolute_integral'],8*np.pi)
    assert abs(constant['rc'])<1e-10 and constant['sd']==0
    waveform=4+3*np.sin(phase)+2*np.cos(2*phase);v,mu=current_parameters(t,waveform,0.,1.)
    assert np.isclose(v['a1'],3.) and np.isclose(v['a2'],2.)
    assert np.isclose(v['rms']**2,mu**2+v['sd']**2)
    checks.append('Analytical constant, harmonic amplitudes and RMS identity')
    # Joint directional gains are invariant under one additive video offset.
    x=np.sin(phase);design=np.c_[np.ones(n),np.where(x>0,x,0),np.where(x<0,x,0)]
    y=5+.8*design[:,1]+.4*design[:,2]
    a=np.linalg.lstsq(design,y,rcond=None)[0];b=np.linalg.lstsq(design,y+27,rcond=None)[0]
    assert np.allclose(a[1:],b[1:]) and np.allclose(a[1:],[.8,.4])
    residual=y-x;centred=residual-residual.mean();shifted=residual+27-(residual+27).mean()
    assert np.allclose(centred,shifted)
    checks.append('Shared-intercept gain and centred residual offset invariance')
    c=pd.read_csv(out/'cycle_features.csv');q=pd.read_csv(out/'condition_features.csv');features=list(definitions().feature)
    assert len(c)==216 and len(q)==72 and c.run.nunique()==4
    assert not c.duplicated(['run','condition','cycle']).any()
    assert np.isfinite(c[features]).all().all()
    for m in range(4):assert np.allclose(c[f'M{m}_rms']**2,c[f'M{m}_arithmetic_mean']**2+c[f'M{m}_sd']**2)
    assert (c.groupby(['run','condition']).size()==3).all()
    assert c.current_n.sum()==25196 and c.video_n.sum()==7555
    assert (c.current_coverage>=.8).all() and (c.video_coverage>=.8).all()
    checks.append('Complete dataset, unique cycles, native sample counts and per-motor RMS identities')
    p=pd.read_csv(out/'loocv_predictions.csv');m=pd.read_csv(out/'loocv_metrics.csv');fold=pd.read_csv(out/'fold_inventory.csv');beta=pd.read_csv(out/'lda_coefficients.csv')
    keys=['window_cycles','strategy','fit_range','model'];fkeys=keys+['held_out_run']
    assert len(fold)==3616 and len(p)==65088 and len(m)==3164
    for row in fold.itertuples():assert row.held_out_run not in row.training_runs.split(';') and len(row.training_runs.split(';'))==3
    assert not p.duplicated(keys+['run','condition','cycle']).any()
    # Rebuild every saved decision score from raw coefficients and feature values.
    coefficient_groups=dict(tuple(beta.groupby(fkeys,sort=False)))
    for values,g in p.groupby(fkeys,sort=False):
        a=coefficient_groups[values]
        source=c if values[0]==1 else q;source=source[source.run==values[-1]]
        if values[1]=='amplitude_specific':source=source[source['range']==values[2]]
        ids=['condition','cycle'] if values[0]==1 else ['condition']
        x=source.set_index(ids).loc[pd.MultiIndex.from_frame(g[ids]) if len(ids)>1 else g.condition,a.feature].to_numpy()
        z=x@a.raw_coefficient.to_numpy()+a.intercept.iloc[0]
        assert np.allclose(z,g.score,rtol=1e-7,atol=1e-7)
    prediction_groups=dict(tuple(p.groupby(keys,sort=False)))
    for row in m.itertuples():
        g=prediction_groups[(row.window_cycles,row.strategy,row.fit_range,row.model)]
        if row.evaluated_range!='all':g=g[g['range']==row.evaluated_range]
        result=counts(g.true_class,g.score,row.score_offset)
        for key in ['tp','fn','fp','tn']:assert result[key]==getattr(row,key)
    checks.append('All 3616 folds exclude the test run; all scores and 3164 metric rows reconstructed')
    # PNG file integrity.
    from PIL import Image
    for path in (out/'figures').rglob('*.png'):
        with Image.open(path) as im:im.load()
    checks.append('All PNG image files decode')
    print('Verified all saved folds, scores and counts.',flush=True)
    result=dict(status='PASS',checks=checks,native_current_samples=int(c.current_n.sum()),native_video_frames=int(c.video_n.sum()),
                observed_runs=4,cycles=216,condition_observations=72,folds=len(fold),held_out_predictions=len(p))
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'outputs');args=a.parse_args();verify(args.output_dir)
