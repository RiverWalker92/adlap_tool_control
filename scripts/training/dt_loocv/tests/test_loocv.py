import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

EVALUATOR=Path(__file__).resolve().parents[1]/'dt_prediction_loocv_v7.py'
spec=importlib.util.spec_from_file_location('evaluator',EVALUATOR)
ev=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=ev
spec.loader.exec_module(ev)

class SpyRegressor(BaseEstimator,RegressorMixin):
    seen=[]
    def __init__(self, offset=0.):
        self.offset=offset
    def fit(self,X,y):
        self.__class__.seen.append(np.asarray(X).copy())
        self.mean=float(np.mean(y))
        return self
    def predict(self,X):
        return np.full(len(X),self.mean+self.offset)

class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root=Path(self.tmp.name).resolve()
        self.files=[]
        self.loads=[]
        for dof in (1,2):
            folder=self.root/'motors'/f'DOF {dof}'
            folder.mkdir(parents=True)
            for run in (1,2,3):
                path=folder/f'run{run}_ros_log.jsonl'
                path.write_text(json.dumps({'id':dof*10+run}))
                self.files.append(path)
        self.module=types.SimpleNamespace(RUN_SPLIT_SEED=42)
        def load(file_paths,**kwargs):
            self.loads.append([str(p) for p in file_paths])
            return [{'source_file':str(p),'id':json.loads(p.read_text())['id']} for p in file_paths]
        def build(segments,**kwargs):
            samples={0:[],1:[]}
            for segment in segments:
                for motor in samples:
                    n=4+motor
                    X=np.column_stack([np.full(n,segment['id']),np.arange(n)])
                    y=np.full(n,segment['id']+motor)
                    meta={'source_file':segment['source_file'],'protocol':segment['split_group'],
                          'dataset_split':segment['dataset_split'],'dynamic_mask':np.arange(n)%2==0,
                          'stationary_mask':np.arange(n)%2!=0}
                    samples[motor].append((X,X.copy(),y,y*2,meta))
            return samples
        def stack(samples):
            return tuple(np.concatenate([s[i] for s in samples]) for i in range(4))
        def masks(samples):
            return tuple(np.concatenate([s[4][name] for s in samples]) for name in ('dynamic_mask','stationary_mask'))
        def by_protocol(samples,prediction,response_kind):
            grouped={}
            offset=0
            for sample in samples:
                y=sample[2 if response_kind=='encoder' else 3]
                p=prediction[offset:offset+len(y)]
                offset+=len(y)
                entry=grouped.setdefault(sample[4]['protocol'],{'y':[],'p':[],'d':[]})
                entry['y'].extend(y);entry['p'].extend(p);entry['d'].extend(sample[4]['dynamic_mask'])
            result={}
            for name,v in grouped.items():
                y=np.array(v['y']);p=np.array(v['p']);d=np.array(v['d'])
                result[name]={'by_phase':{'overall':ev.metrics(y,p),'dynamic':ev.metrics(y[d],p[d])}}
            return result
        def select(tasks):
            names=sorted(tasks[0]['results'])
            votes={n:0 for n in names};ranks={n:0 for n in names}
            for task in tasks:
                ranked=sorted(names,key=lambda n:(task['score_function'](task['results'][n]),n))
                votes[ranked[0]]+=1
                for rank,n in enumerate(ranked,1):ranks[n]+=rank
            choice=min(names,key=lambda n:(-votes[n],ranks[n]/len(tasks),n))
            return choice,{'policy':'majority_vote_over_motor_output_task_winners','vote_counts':votes}
        self.module.load_segments_from_files=load
        self.module.build_dataset=build
        self.module.get_split_group=lambda segment,*args:Path(segment['source_file']).parent.name
        self.module.stack_samples=stack
        self.module.stack_evaluation_masks=masks
        self.module.make_encoder_models=lambda:{'ridge':SpyRegressor(),'random_forest':SpyRegressor(2.)}
        self.module.make_current_models=self.module.make_encoder_models
        self.module.compute_metrics_by_protocol=by_protocol
        self.module.compute_current_zone_metrics=lambda y,p,d,s:{'moving_current_mA':ev.metrics(y[d],p[d])}
        self.module.encoder_model_selection_score=lambda r:float(np.mean([v['by_phase']['dynamic']['rmse'] for v in r['by_protocol'].values()]))
        self.module.current_model_selection_score=lambda r:float(np.mean([v['by_phase']['dynamic']['mae'] for v in r['by_protocol'].values()]))
        self.module.select_common_model_family=select
        self.features=[f'gearbox_{i}' for i in range(6)]+['cmd_instr_1']
        rows=[]
        for trial,correction in [('001',2.),('002',5.),('003',10.)]:
            for t in range(6):
                rows.append(dict(trial_id=trial,time=float(t),video_angle=float(t)+correction,dt_prediction=float(t),
                                 prediction_error=-999.,**{f:float(int(trial))+t*.01 for f in self.features}))
        self.df=pd.DataFrame(rows)
        self.csv=self.root/'bend_training_dataset.csv'
        self.df.to_csv(self.csv,index=False)
        def feature_select(df):
            missing=set(self.features)-set(df)
            if missing:raise RuntimeError(f'missing {missing}')
            return self.features
        self.hybrid=types.SimpleNamespace(select_feature_columns=feature_select,
            make_models=lambda:{'ridge':SpyRegressor(),'random_forest':SpyRegressor(2.)})
        self.job=dict(name='motor_only_targets',coupling_mode='motor_only',input_source='targets',variant='',instrument='',files=self.files)
    def tearDown(self):self.tmp.cleanup()
    def test_motor_outer_run_excluded_from_all_inner_and_final_fits(self):
        held=self.files[0];held_id=json.loads(held.read_text())['id']
        folder=self.root/'fold';folder.mkdir()
        SpyRegressor.seen=[]
        rows=ev.motor_fold(self.module,self.job,held,'nested','ridge',['ridge','random_forest'],folder)
        self.assertEqual(len(rows),12) # two motors, two outputs, three phases
        self.assertTrue(SpyRegressor.seen)
        for X in SpyRegressor.seen:self.assertNotIn(held_id,X[:,0])
        report=json.loads((folder/'selection.json').read_text())
        self.assertNotIn(str(held),report['outer_training_files'])
        self.assertNotIn(str(held),report['inner_validation_file_by_protocol'].values())
        self.assertEqual(self.loads[0],list(map(str,self.files[1:])))
        self.assertEqual(self.loads[1],[str(held)])
        for row in rows:self.assertNotIn(str(held),json.loads(row['training_files_json']))
    def test_inner_split_is_protocol_balanced_and_whole_run(self):
        samples=ev.prepare_motor(self.module,self.files,'motor_only','targets','train')
        train,val,held=ev.inner_motor_split(samples,42)
        self.assertEqual(len(held),2)
        for motor in train:
            a={ev.source_of(s) for s in train[motor]};b={ev.source_of(s) for s in val[motor]}
            self.assertFalse(a&b)
            self.assertEqual(len(a),4);self.assertEqual(len(b),2)
    def test_insufficient_inner_runs_rejected(self):
        samples=ev.prepare_motor(self.module,self.files[:1],'motor_only','targets','train')
        with self.assertRaisesRegex(RuntimeError,'at least three'):ev.inner_motor_split(samples,42)
    def test_fixed_family_requires_no_inner_split(self):
        job=dict(self.job,files=self.files[:2])
        folder=self.root/'fixed';folder.mkdir()
        rows=ev.motor_fold(self.module,job,self.files[0],'fixed','ridge',['ridge'],folder)
        self.assertTrue(rows)
        self.assertEqual({r['model_family'] for r in rows},{'ridge'})
    def test_metric_formulas_and_nonfinite_rejection(self):
        result=ev.metrics([1.,2.],[0.,0.])
        self.assertEqual(result['bias'],1.5)
        self.assertAlmostEqual(result['rmse'],np.sqrt(2.5))
        with self.assertRaises(ValueError):ev.metrics([1.,np.nan],[0.,0.])
    def test_pooling_correct_rmse_and_maximum(self):
        base=dict(dataset='d',configuration='m',input_source='targets',variant='',instrument_config='',run_id='r',held_out_file='f',protocol='p',model_family='ridge',output='current',unit='mA',phase='overall')
        rows=[dict(base,n_samples=1,mae=1.,rmse=1.,bias=1.,max_abs_error=1.),dict(base,n_samples=3,mae=3.,rmse=3.,bias=3.,max_abs_error=3.)]
        row=ev.pool_motor_runs(rows)[0]
        self.assertEqual(row['mae'],2.5);self.assertAlmostEqual(row['rmse'],np.sqrt(7.));self.assertEqual(row['max_abs_error'],3.)
    def test_same_basename_different_sources_remain_distinct(self):
        folder=self.root/'fold';folder.mkdir()
        rows=ev.motor_fold(self.module,self.job,self.files[0],'fixed','ridge',['ridge'],folder)
        extra=[dict(r,held_out_file='other',run_id=rows[0]['run_id']) for r in rows]
        self.assertEqual(len(ev.pool_motor_runs(rows+extra)),12)
    def test_hybrid_exact_seven_inputs_and_id_preservation(self):
        df,features,origin=ev.load_hybrid_data(self.hybrid,{},self.root,self.csv,None)
        self.assertEqual(features,self.features)
        self.assertEqual(sorted(df['trial_id'].unique()),['001','002','003'])
        self.assertTrue((df['prediction_error']>0).all())
        old=df.drop(columns=['cmd_instr_1']);path=self.root/'old.csv';old.to_csv(path,index=False)
        with self.assertRaises(RuntimeError):ev.load_hybrid_data(self.hybrid,{},self.root,path,None)
    def test_hybrid_outer_trial_exclusion_and_original_angle_reference(self):
        df,features,_=ev.load_hybrid_data(self.hybrid,{},self.root,self.csv,None)
        SpyRegressor.seen=[]
        out=self.root/'hybrid';out.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            rows,_=ev.evaluate_hybrid(self.hybrid,df,features,'nested','ridge',['ridge','random_forest'],out)
        # Two candidate inner fits + one final fit for each fold.
        self.assertEqual(len(SpyRegressor.seen),9)
        for i,trial in enumerate(['001','002','003']):
            for X in SpyRegressor.seen[i*3:(i+1)*3]:self.assertFalse(np.any(np.isclose(X[:,0],float(int(trial)))))
            report=json.loads((out/'hybrid_folds'/f'fold_{i+1:03d}'/'selection.json').read_text())
            self.assertNotIn(trial,report['outer_training_trials'])
            self.assertNotEqual(trial,report['inner_validation_trial'])
        a=next(r for r in rows if r['model']=='physics' and r['run_id']=='001')
        self.assertAlmostEqual(a['bias'],np.deg2rad(2.)) # no new zeroing
    def test_real_hybrid_random_forest_factory(self):
        self.hybrid.make_models=lambda:{'random_forest':RandomForestRegressor(n_estimators=5,min_samples_leaf=1,random_state=42)}
        out=self.root/'realrf';out.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            rows,normalized=ev.evaluate_hybrid(self.hybrid,self.df,self.features,'fixed','random_forest',['random_forest'],out)
        self.assertEqual(len(rows),6);self.assertEqual(len(normalized),6)
        self.assertTrue(np.isfinite([r['mae'] for r in rows]).all())
    def test_training_paths_rebase_old_root_to_v7(self):
        native={'test_data_root':'/data','training_data_root':'trainings data V6'}
        p=ev.training_path(native,self.root,'/data/trainings data V6/motors')
        self.assertEqual(p,self.root/'motors')
        with self.assertRaises(ValueError):ev.training_path(native,self.root,'/unrelated/data')
    def test_native_jobs_keep_sources_variants_and_instruments_separate(self):
        native={'motor_dt':{'configurations':{
            'motor_only_commands':{'input_dir':'commands'},'motor_only_targets':{'input_dir':'targets'},
            'gearbox_only':{'input_dir':'gearboxes'},'full_setup':{'input_dir':'full_setup'}}}}
        for folder in ['commands','targets','gearboxes/gearbox_1','gearboxes/gearbox_2','full_setup/gripper','full_setup/scissors']:
            p=self.root/folder;p.mkdir(parents=True)
            for i in (1,2,3):(p/f'run{i}.jsonl').write_text('{}')
        self.module.find_motor_jsonl_files=lambda input_dir,coupling_mode:list(input_dir.glob('*.jsonl'))
        self.module.load_available_gearbox_variants=lambda p:(['gearbox_1','gearbox_2'],'gearbox_2')
        jobs,skipped=ev.resolve_jobs(self.module,native,self.root,self.root/'config.yaml')
        self.assertEqual(len(jobs),6);self.assertFalse(skipped)
        self.assertEqual({j['instrument'] for j in jobs if j['coupling_mode']=='full_setup'},{'gripper','scissors'})
        self.assertEqual(len({str(p) for j in jobs for p in j['files']}),18)
    def test_summaries_equal_weight_runs_and_keep_datasets(self):
        rows=[dict(dataset='a',mae=1.,rmse=1.,bias=1.,max_abs_error=1.),dict(dataset='a',mae=3.,rmse=3.,bias=3.,max_abs_error=3.),dict(dataset='b',mae=100.,rmse=100.,bias=100.,max_abs_error=100.)]
        result=ev.summarize(rows,['dataset'])
        self.assertEqual(result[0]['mae_mean'],2.)
        self.assertAlmostEqual(result[0]['mae_sd'],np.sqrt(2.))
        self.assertIsNone(result[1]['mae_sd'])
    def test_cli_help(self):
        result=subprocess.run([sys.executable,str(EVALUATOR),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0)
        self.assertIn('--selection',result.stdout)
    def test_complete_pipeline_with_native_yaml_and_exports(self):
        cfg=self.root/'training_config.yaml'
        cfg.write_text('motor_dt:\n  configurations:\n    motor_only_targets:\n      input_dir: motors\nhybrid_bend_dt:\n  input_dir: hybrid_raw\n')
        gearbox=self.root/'gearbox_params.yaml';gearbox.write_text('{}')
        self.module.DEFAULT_TRAINING_CONFIG_PATH=cfg
        self.module.DEFAULT_GEARBOX_CONFIG_PATH=gearbox
        self.module.find_motor_jsonl_files=lambda input_dir,coupling_mode:list(input_dir.rglob('*.jsonl'))
        self.hybrid.DEFAULT_TRAINING_CONFIG_PATH=cfg
        self.hybrid.build_dataset=lambda folder:self.df.copy()
        sources={kind:self.root/f'train_{kind}.py' for kind in ['motor','hybrid']}
        for path in sources.values():path.write_text('# synthetic training interface fixture\n')
        original_import,original_find,original_argv=ev.import_script,ev.find_training_script,sys.argv
        ev.find_training_script=lambda explicit,kind:sources[kind]
        ev.import_script=lambda path,name,required:self.module if 'motor' in name else self.hybrid
        out=self.root/'complete'
        sys.argv=[str(EVALUATOR),'--data-root',str(self.root),'--models','ridge','random_forest','--output-dir',str(out)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):ev.main()
        finally:
            ev.import_script,ev.find_training_script,sys.argv=original_import,original_find,original_argv
        self.assertEqual(len(pd.read_csv(out/'motor_per_motor_metrics.csv')),72)
        self.assertEqual(len(pd.read_csv(out/'motor_per_run_metrics.csv')),36)
        self.assertEqual(len(pd.read_csv(out/'instrument_per_run_metrics.csv')),6)
        self.assertEqual(json.loads((out/'evaluation_metadata.json').read_text())['selection'],'nested')
        self.assertEqual(len(list((out/'motor_folds').rglob('selection.json'))),6)
        self.assertEqual(len(list((out/'hybrid_folds').rglob('selection.json'))),3)
        text=(out/'dt_performance_tables.tex').read_text()
        self.assertIn('motor\\_only\\_targets',text)
        self.assertNotIn('DOF4',text)

if __name__=='__main__':unittest.main(verbosity=2)
