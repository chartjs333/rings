import ctypes as ct
import math
import tempfile
import unittest
from pathlib import Path
from native import NativeClassifier,FrameBuffer
from multiclass import MulticlassClassifier,MulticlassResults

class MulticlassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory(dir='work');root=Path(cls.folder.name)
        cls.features=root/'features.txt';cls.head=root/'head.txt'
        lines=['RINGS2D_CPP_V3',f'half_width {math.pi/2:.17g}','edge_weight 0.4','color_weight 0.25',
               f'velocity {math.pi/2:.17g}','phase_gain 0.25','readout_bins 2']
        for name,values in (('mean',[.5]*768),('scale',[.2]*768),('angles',[0.1*i for i in range(768)]),('weights',[0]*1536)):
            lines.extend([f'{name} {len(values)}',' '.join(map(str,values))])
        lines.append('bias 0');cls.features.write_text('\n'.join(lines)+'\n')
        cls.weights=[((i*3)%17-8)*.001 for i in range(15360)];cls.bias=[k*.01 for k in range(10)]
        cls.head.write_text('RINGS_CIFAR10_READOUT_V1\nclasses 10\nfeatures 1536\nweights 15360\n'+' '.join(map(str,cls.weights))+'\nbiases 10\n'+' '.join(map(str,cls.bias))+'\n')
    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def test_feature_scores_and_argmax(self):
        stride=3104;frames=FrameBuffer(bytearray(i%251 for i in range(stride*7)),count=7,stride=stride)
        for reset in (True,False):
            with NativeClassifier(self.features) as extractor,MulticlassClassifier(self.features,self.head) as model:
                features=(ct.c_double*(7*1536))();extractor.features_into(frames,features,reset_each=reset)
                output=MulticlassResults(7);model.predict_into(frames,output,reset_each=reset)
                for n in range(7):
                    expected=[math.fsum(features[n*1536+j]*self.weights[k*1536+j] for j in range(1536))+self.bias[k] for k in range(10)]
                    for k in range(10):self.assertAlmostEqual(output.scores[n*10+k],expected[k],places=12)
                    self.assertEqual(output.labels[n],max(range(10),key=lambda k:expected[k]))

    def test_batch_single_reset_and_stream(self):
        frames=FrameBuffer(bytearray(i%251 for i in range(3072*7)))
        with MulticlassClassifier(self.features,self.head) as model:
            for reset in (True,False):
                model.reset();batch=MulticlassResults(7);model.predict_into(frames,batch,reset_each=reset)
                model.reset()
                for i in range(7):
                    one=FrameBuffer(frames.view[i*3072:(i+1)*3072]);out=MulticlassResults(1)
                    model.predict_into(one,out,reset_each=reset)
                    self.assertEqual(list(out.scores),list(batch.scores)[i*10:(i+1)*10]);self.assertEqual(out.labels[0],batch.labels[i])

    def test_empty_and_small_output(self):
        with MulticlassClassifier(self.features,self.head) as model:
            model.predict_into(FrameBuffer(bytearray()),MulticlassResults(0))
            with self.assertRaises(RuntimeError):model.predict_into(FrameBuffer(bytearray(3072)),MulticlassResults(0))
            with self.assertRaises(ValueError):model.predict_into(FrameBuffer(bytearray(3072)),MulticlassResults(1),reset_each=2)

    def test_invalid_model(self):
        bad=Path(self.folder.name)/'bad.txt';bad.write_text(self.head.read_text().replace('classes 10','classes 11'))
        with self.assertRaises(ValueError):MulticlassClassifier(self.features,bad)

    def test_bounds_before_writes(self):
        frames=FrameBuffer(bytearray(3072));out=MulticlassResults(1);out.scores[0]=123;out.labels[0]=45
        with MulticlassClassifier(self.features,self.head) as model:
            for count,byte_count,stride,score_count in ((1,3071,3072,10),(1,3072,3071,10),(1,3072,3072,9),((1<<64)-1,3072,3072,10)):
                self.assertEqual(model.lib.mc_process(model.handle,frames.pointer,byte_count,count,stride,1,out.scores,score_count,out.labels,1),-1)
                self.assertEqual((out.scores[0],out.labels[0]),(123,45))

if __name__=='__main__':unittest.main()
