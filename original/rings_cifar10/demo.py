from pathlib import Path
import numpy as np
from multiclass import MulticlassClassifier,MulticlassResults
from native import FrameBuffer
from dataset import NAMES

root=Path(__file__).resolve().parent
with np.load(root/'data/examples.npz',allow_pickle=False) as saved:images=saved['images'][:10].copy();truth=saved['truth'][:10]
with MulticlassClassifier(root/'data/phase_features.txt',root/'data/phase_readout.txt') as model:
    result=MulticlassResults(len(images));model.predict_into(FrameBuffer(images),result)
for i in range(len(images)):print(f'{i}: true={NAMES[int(truth[i])]}, predicted={NAMES[result.labels[i]]}')
