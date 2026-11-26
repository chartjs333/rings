"""Full CIFAR-10, native ring features, validation-selected Ridge and MLP."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,ctypes as ct,hashlib,json,math,platform,time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from native import NativeClassifier,FrameBuffer
from multiclass import MulticlassClassifier,MulticlassResults
from dataset import load_cifar,preprocess,NAMES
from train_mlp import fit as fit_mlp,predict as predict_mlp

ROOT=Path(__file__).resolve().parent
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def feature_model(path,mean,scale,angles,gain):
    lines=['RINGS2D_CPP_V3',f'half_width {math.pi/2:.17g}','edge_weight 0.40000000000000002',
           'color_weight 0.25',f'velocity {math.pi/2:.17g}',f'phase_gain {gain:.17g}','readout_bins 2']
    for name,values in (('mean',mean),('scale',scale),('angles',angles),('weights',np.zeros(1536))):
        lines.extend([f'{name} {len(values)}',' '.join(format(float(v),'.17g') for v in values)])
    lines.append('bias 0');path.write_text('\n'.join(lines)+'\n')
def readout_model(path,clf):
    assert clf.coef_.shape==(10,1536) and np.array_equal(clf.classes_,np.arange(10))
    lines=['RINGS_CIFAR10_READOUT_V1','classes 10','features 1536','weights 15360',
           ' '.join(format(float(v),'.17g') for v in clf.coef_.ravel()),'biases 10',
           ' '.join(format(float(v),'.17g') for v in clf.intercept_)]
    path.write_text('\n'.join(lines)+'\n')
def features(path,images,target):
    out=np.lib.format.open_memmap(target,mode='w+',dtype=np.float64,shape=(len(images),1536))
    with NativeClassifier(path) as model:
        for start in range(0,len(images),512):
            batch=images[start:start+512];values=(ct.c_double*(len(batch)*1536))()
            model.features_into(FrameBuffer(batch),values)
            out[start:start+len(batch)]=np.ctypeslib.as_array(values).reshape(len(batch),1536)
    out.flush();return out
def native_predict(feature_path,readout_path,images):
    result=MulticlassResults(len(images))
    with MulticlassClassifier(feature_path,readout_path) as model:model.predict_into(FrameBuffer(images),result)
    return np.array(result.scores).reshape(-1,10),np.array(result.labels)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',required=True,type=Path);args=parser.parse_args()
    out=ROOT/'results';work=ROOT/'work';data_dir=ROOT/'data'
    if (out/'accuracy.json').exists():raise RuntimeError('Results already exist; use a fresh output project')
    protocol=dict(dataset='Full official CIFAR-10',train=50000,test=10000,inner=40000,validation=10000,
                  split_seed=42,phase_gains=[0,.25],ridge_alphas=[10.,100.,1000.,10000.],
                  fixed=dict(bins=2,edge=.4,color=.25,steps=4,half_width=math.pi/2,phase_seed=0),
                  mlp=dict(layers=[768,64,10],epochs_grid=[10,20,40],optimizer='Adam',lr=.001,weight_decay=.0001,batch=256,seed=0),
                  selection='ridge: validation correct then larger alpha; MLP: validation correct then fewer epochs',
                  preprocessing='32x32 RGB -> 2x2 mean ->16x16 RGB; training-only standardization',
                  classes=NAMES,ring_readout_parameters=15370,mlp_parameters=49866,
                  caveat='2000 cat/dog test images already appeared in earlier experiments; not a wholly new independent benchmark',
                  native_core_sha256=hashlib.sha256((ROOT/'rings.hpp').read_bytes()).hexdigest())
    save(out/'protocol_before_fit.json',protocol)
    dataset=load_cifar(args.archive);train,y=dataset['train'];test,truth=dataset['test']
    inner,val=train_test_split(np.arange(50000),test_size=.2,stratify=y,random_state=42)
    np.savez_compressed(out/'split.npz',inner=inner,validation=val)
    angles=np.array(json.loads((ROOT/'reference/initial_angles.json').read_text())['angles'])
    pooled=preprocess(train,np.float64)
    scaler=StandardScaler().fit(pooled[inner]);rows=[];selected={};started=time.perf_counter()
    for name,gain in (('linear',0.),('phase',.25)):
        template=work/(name+'_template.txt');feature_model(template,scaler.mean_,scaler.scale_,angles,gain)
        f=features(template,train,work/'train_features.npy')
        for alpha in protocol['ridge_alphas']:
            t=time.perf_counter();clf=RidgeClassifier(alpha=alpha,solver='cholesky').fit(f[inner],y[inner])
            correct=int(np.count_nonzero(clf.predict(f[val])==y[val]))
            row=dict(model=name,phase_gain=gain,alpha=alpha,correct=correct,total=len(val),fit_predict_seconds=time.perf_counter()-t)
            rows.append(row);print('RIDGE validation',json.dumps(row),flush=True)
            del clf
        selected[name]=sorted([r for r in rows if r['model']==name],key=lambda r:(-r['correct'],-r['alpha']))[0]
        del f
    _,history=fit_mlp(train[inner],y[inner],40,(train[val],y[val]))
    selected['mlp']=sorted(history,key=lambda r:(-r['correct'],r['epochs']))[0]
    save(out/'selected_before_test.json',selected);save(out/'validation.json',dict(ridge=rows,mlp=history))
    scaler=StandardScaler().fit(pooled);del pooled
    models={};fit_times={}
    for name,gain in (('linear',0.),('phase',.25)):
        feature_path=data_dir/(name+'_features.txt');readout_path=data_dir/(name+'_readout.txt')
        feature_model(feature_path,scaler.mean_,scaler.scale_,angles,gain)
        t=time.perf_counter();f=features(feature_path,train,work/'train_features.npy');extraction=time.perf_counter()-t
        t=time.perf_counter();clf=RidgeClassifier(alpha=selected[name]['alpha'],solver='cholesky').fit(f,y);fitting=time.perf_counter()-t
        readout_model(readout_path,clf);models[name]=(feature_path,readout_path,clf)
        fit_times[name]=dict(extraction_seconds=extraction,readout_fit_seconds=fitting);del f
    t=time.perf_counter();mlp,_=fit_mlp(train,y,selected['mlp']['epochs']);fit_times['mlp']=dict(fit_seconds=time.perf_counter()-t)
    np.savez_compressed(data_dir/'mlp.npz',**mlp);total_fit=time.perf_counter()-started
    # Test consulted only after all selection and final fitting are finished.
    predictions={};scores={};metrics={}
    for name,(feature_path,readout_path,clf) in models.items():
        scores[name],predictions[name]=native_predict(feature_path,readout_path,test)
        f=features(feature_path,test,work/'test_features.npy');reference=clf.decision_function(f)
        np.testing.assert_allclose(scores[name],reference,rtol=1e-10,atol=1e-11)
        np.testing.assert_array_equal(predictions[name],clf.predict(f));del f
        metrics[name]=dict(parameters=15370,max_native_sklearn_score_error=float(np.max(np.abs(scores[name]-reference))))
    scores['mlp']=predict_mlp(mlp,test);predictions['mlp']=scores['mlp'].argmax(1);metrics['mlp']=dict(parameters=49866)
    for name,pred in predictions.items():
        confusion=np.zeros((10,10),dtype=int);np.add.at(confusion,(truth,pred),1)
        metrics[name].update(correct=int(np.count_nonzero(pred==truth)),total=10000,accuracy=float(np.mean(pred==truth)),confusion=confusion.tolist())
    paired={}
    for name in ('linear','mlp'):
        gained=int(np.count_nonzero((predictions['phase']==truth)&(predictions[name]!=truth)))
        lost=int(np.count_nonzero((predictions['phase']!=truth)&(predictions[name]==truth)))
        paired[name]=dict(gained=gained,lost=lost,p_two_sided=float(binomtest(gained,gained+lost,.5).pvalue) if gained+lost else 1.)
    save(out/'accuracy.json',dict(metrics=metrics,paired=paired,selected=selected,fit_times=fit_times,total_selection_and_fit_seconds=total_fit,host=platform.platform()))
    np.savez_compressed(out/'predictions.npz',truth=truth,**predictions,**{name+'_scores':s for name,s in scores.items()})
    with (out/'predictions.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['official_test_id','truth','phase','linear','mlp'])
        writer.writerows(zip(range(10000),truth,predictions['phase'],predictions['linear'],predictions['mlp']))
    np.savez_compressed(data_dir/'examples.npz',images=test[:128],truth=truth[:128])
    print(json.dumps({'metrics':metrics,'paired':paired,'selected':selected},indent=2),flush=True)
if __name__=='__main__':main()
