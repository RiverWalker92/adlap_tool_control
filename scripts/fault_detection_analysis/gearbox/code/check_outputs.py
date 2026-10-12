#!/usr/bin/env python3
"""Verify slope arithmetic, recorded profiles, validation and report outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyse import COLUMNS, PARAMETERS, residual_phase_slope, extract_parameters, observation_windows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figures',type=Path,required=True)
    args=parser.parse_args();output=args.output
    phase=np.linspace(-np.pi,np.pi,151)[:-1]
    np.testing.assert_allclose(residual_phase_slope(2.3*phase+5,phase),2.3,atol=1e-12)
    np.testing.assert_allclose(residual_phase_slope(np.full_like(phase,5),phase),0,atol=1e-12)
    features=pd.read_csv(output/'features.csv')
    profiles=np.load(output/'rotation_profiles.npz')
    grid=profiles['phase']
    assert len(features)==48 and len(COLUMNS)==24 and len(PARAMETERS)==11
    assert len(grid)==151 and np.isfinite(features[COLUMNS].to_numpy()).all()
    for run in features.run.unique():
        assert profiles[run].shape==(30,151,2)
        for window,start,end in observation_windows():
            profile=profiles[run][start:end].mean(axis=0)
            row=features[(features.run==run)&(features.window==window)].iloc[0]
            expected=extract_parameters(profile,grid)
            for column,value in expected.items():
                np.testing.assert_allclose(row[column],value,rtol=1e-11,atol=1e-11)
            for index,motor in enumerate(('M0','M3')):
                direct=np.linalg.lstsq(np.column_stack([grid[:-1],np.ones(150)]),
                                       profile[:-1,index],rcond=None)[0][0]
                np.testing.assert_allclose(row[motor+'_rc'],direct,atol=1e-12)
    definitions=json.loads((output/'model_definitions.json').read_text())
    assert len(definitions)==35 and all(len(d['features'])<=2 for d in definitions.values())
    cv=pd.read_csv(output/'cv_summary.csv')
    assert len(cv)==70
    rms=cv[cv.model=='M0_M3_rms']
    assert (rms.sensitivity==1).all() and (rms.specificity==1).all()
    assert list(rms.sort_values('n').TP)==[18,9] and list(rms.sort_values('n').TN)==[108,27]
    predictions=pd.read_csv(output/'cv_predictions.csv')
    for (n,model),group in predictions.groupby(['n','model']):
        assert len(group)==(126 if n==5 else 36)
    for scope in ('pooled','healthy','degraded'):
        matrix=pd.read_csv(output/f'correlation_{scope}.csv',index_col=0).to_numpy()
        assert matrix.shape==(24,24)
        np.testing.assert_allclose(matrix,matrix.T,atol=1e-12)
    for name in ('boxplots_M0.png','boxplots_M3_joint.png','correlation_pooled.png',
                 'correlation_healthy.png','correlation_degraded.png','lda_rms.png',
                 'rc_final_block_fit.png'):
        assert (args.figures/name).stat().st_size>1000
    print('RC verified against an independent least squares fit for every profile.')
    print('All profiles, 35 direct classifiers, validation counts and report outputs checked.')


if __name__=='__main__':main()
