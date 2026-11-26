"""Full official CIFAR-10 binary archive. No pickle and no archive extraction."""
import hashlib
import tarfile
from pathlib import Path
import numpy as np

MD5='c32a1d4ab5d03f1284b67883e8d87530'
NAMES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def load_cifar(path):
    with Path(path).open('rb') as f:
        if hashlib.file_digest(f,'md5').hexdigest()!=MD5:raise ValueError('CIFAR checksum mismatch')
    result={}
    with tarfile.open(path,'r:gz') as archive:
        names=archive.extractfile('cifar-10-batches-bin/batches.meta.txt').read().decode().strip().splitlines()
        assert names==NAMES
        for split,files in (('train',[f'data_batch_{i}.bin' for i in range(1,6)]),('test',['test_batch.bin'])):
            images=[];labels=[]
            for name in files:
                data=archive.extractfile('cifar-10-batches-bin/'+name).read();assert len(data)==10000*3073
                rows=np.frombuffer(data,dtype=np.uint8).reshape(10000,3073)
                images.append(rows[:,1:].reshape(-1,3,32,32).transpose(0,2,3,1).copy())
                labels.append(rows[:,0].astype(np.int64))
            x,y=np.concatenate(images),np.concatenate(labels)
            assert np.array_equal(np.bincount(y),[len(y)//10]*10)
            result[split]=(x,y)
    return result

def preprocess(images,dtype=np.float32):
    return images.astype(dtype).reshape(-1,16,2,16,2,3).mean(axis=(2,4)).reshape(-1,768)/255.
