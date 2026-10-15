"""Apply a fixed exported LDA to individual new cuts; never average cuts."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def predict(df,models,scenario='healthy_all',selection='later',model='directions_M3_phase_mean',weight=1):
 f=next(m for m in models if m['scenario']==scenario and m['selection']==selection and m['model']==model)
 cols=f['parameters'];co=np.array(f['coefficients']);b=np.array(f['intercept'])
 cuts={'all':[1,2,3],'first':[1],'later':[2,3]}[selection]
 d=df[df.cut.isin(cuts)].copy();assert np.isfinite(d[cols].to_numpy()).all()
 z=d[cols].to_numpy()@co.T+b
 if len(f['class_labels'])==2:
  d['lda_score']=z[:,0];d['alarm_score']=z[:,0]+np.log(weight);d['prediction']=(d.alarm_score>0).astype(int)
 else:
  if weight!=1:raise ValueError('The weight3 reference applies only to binary LDA.')
  for j in range(z.shape[1]):d[f'lda_score_class{j}']=z[:,j]
  d['prediction']=z.argmax(1)
 d['predicted_label']=[f['class_labels'][i] for i in d.prediction]
 return d[['cycle_id','run_id','iteration','run_number','cut']+cols+[c for c in d if c.startswith(('lda_score','alarm_score'))]+['prediction','predicted_label']]
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--models',type=Path,default=ROOT/'outputs/fitted_models.json');ap.add_argument('--scenario',default='healthy_all');ap.add_argument('--selection',choices=['all','first','later'],default='later');ap.add_argument('--model',default='directions_M3_phase_mean');ap.add_argument('--weight',type=int,choices=[1,3],default=1);a=ap.parse_args()
 predict(pd.read_csv(a.input),json.loads(a.models.read_text()),a.scenario,a.selection,a.model,a.weight).to_csv(a.output,index=False)
 print('Predictions per cut written:',a.output)
