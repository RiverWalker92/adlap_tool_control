"""Audit old measured-current results without rerunning or modifying them."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REMOVED_NAMES={'rc30_80','rc_closing_rms'}

def summarize(source=None,destination=None):
    source=Path(source or ROOT/'outputs')
    destination=Path(destination or ROOT/'outputs_residual')
    path=source/'loocv_metrics.csv'
    if not path.exists():
        print('No old outputs/loocv_metrics.csv: old measured-current scores unavailable locally.',flush=True)
        return None
    old=pd.read_csv(path)
    selected=old.model.str.startswith(('current_','filtered_'))|old.model.isin(REMOVED_NAMES)
    table=old[selected & (old.weight==1)].copy()
    for i,r in table.iterrows():
        cm=np.asarray(json.loads(r.confusion_matrix))
        table.loc[i,'alarm_TP']=int(cm[1:,1:].sum())
        table.loc[i,'alarm_FN']=int(cm[1:,0].sum())
        table.loc[i,'alarm_TN']=int(cm[0,0])
        table.loc[i,'alarm_FP']=int(cm[0,1:].sum())
        if len(cm)==2:
            for key,val in zip(['TP','FN','TN','FP'],[cm[1,1],cm[1,0],cm[0,0],cm[0,1]]):
                table.loc[i,key]=int(val)
    destination.mkdir(parents=True,exist_ok=True)
    table.to_csv(destination/'previous_measured_current_models.csv',index=False)
    comparison=old[(old.scenario=='healthy_all')&(old.weight==1)].copy()
    for i,r in comparison.iterrows():
        cm=np.asarray(json.loads(r.confusion_matrix))
        comparison.loc[i,'FN']=int(cm[1,0]);comparison.loc[i,'FP']=int(cm[0,1])
    top=comparison.sort_values(['selection','FN','FP']).groupby('selection').head(5)
    top.to_csv(destination/'previous_best_binary_models.csv',index=False)
    print('Archived measured-current scores:',len(table),'rows; old outputs unchanged.',flush=True)
    return table

if __name__=='__main__':summarize()
