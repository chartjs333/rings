"""One-thread raw RGB to ten-class labels; current deployed implementations."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,random,statistics,time,platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from native import FrameBuffer
from multiclass import MulticlassClassifier,MulticlassResults
from train_mlp import predict

def main():
    root=Path(__file__).resolve().parent
    with np.load(root/'data/examples.npz',allow_pickle=False) as saved:images=saved['images'].copy()
    with np.load(root/'data/mlp.npz',allow_pickle=False) as saved:mlp={k:saved[k].copy() for k in saved.files}
    rng=random.Random(2026);results={};models={name:MulticlassClassifier(root/'data'/f'{name}_features.txt',root/'data'/f'{name}_readout.txt') for name in ('phase','linear')}
    try:
        with threadpool_limits(limits=1):
            for count in (1,128):
                raw=images[:count];frames=FrameBuffer(raw);outputs={name:MulticlassResults(count) for name in models}
                functions={name:(lambda name=name:models[name].predict_into(frames,outputs[name])) for name in models}
                functions['mlp']=lambda:predict(mlp,raw).argmax(1)
                for _ in range(30):
                    for fn in functions.values():fn()
                times={name:[] for name in functions}
                for _ in range(301):
                    names=list(functions);rng.shuffle(names)
                    for name in names:
                        start=time.perf_counter_ns();functions[name]();times[name].append((time.perf_counter_ns()-start)/1000/count)
                results[str(count)]={name:dict(median_us_per_frame=statistics.median(v),p95_us_per_frame=sorted(v)[int(.95*(len(v)-1))],samples_us_per_frame=v) for name,v in times.items()}
            pools=threadpool_info();assert all(p['num_threads']==1 for p in pools)
    finally:
        for model in models.values():model.close()
    report=dict(repetitions=301,warmup=30,order_seed=2026,threads=1,
                frames='count=1 repeats first official test image; count=128 repeats first 128 distinct test images',
                scope='RGB pooling, training-fitted normalization, feature extractor, ten class scores, argmax',
                excluded='model/data loading, initial input/output-buffer setup and result conversion to Python lists',
                caveat='C++ float64 rings and NumPy/BLAS float32 MLP; implementation comparison, not backend-matched architecture comparison',
                threadpools=pools,platform=platform.platform(),results=results)
    (root/'results/latency.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({n:{model:row['median_us_per_frame'] for model,row in rows.items()} for n,rows in results.items()},indent=2))
if __name__=='__main__':main()
