import unittest, importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit.py'));a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class Tests(unittest.TestCase):
 def test_shifted_fp32_ce(self):
  ids=torch.tensor([[0,1,2]]);logits=torch.tensor([[[0.,9.,0.],[0.,0.,9.],[9.,0.,0.]]],dtype=torch.bfloat16)
  expected=F.cross_entropy(logits[:,:-1].float().reshape(-1,3),ids[:,1:].reshape(-1),reduction='none')
  self.assertTrue(torch.equal(a.ce(logits,ids).flatten(),expected));self.assertLess(a.ce(logits,ids).max(),.001)
 def test_topk_then_relu_dense_equivalent(self):
  torch.manual_seed(4);s={'W_enc':torch.randn(7,17),'b_enc':torch.randn(17),'W_dec':torch.randn(17,7),'b_dec':torch.randn(7)};x=torch.randn(9,7)
  r,v,ix=a.reconstruct(x,s,3);pre=(x-s['b_dec'])@s['W_enc']+s['b_enc'];vv,ii=pre.topk(3,dim=-1)
  dense=torch.zeros_like(pre).scatter_(-1,ii,vv.relu())@s['W_dec']+s['b_dec']
  torch.testing.assert_close(r,dense);self.assertTrue(((v>0).sum(-1)<=3).all())
 def test_negative_topk_does_not_fire(self):
  s={'W_enc':torch.zeros(2,4),'b_enc':-torch.ones(4),'W_dec':torch.ones(4,2),'b_dec':torch.tensor([2.,3.])};r,v,_=a.reconstruct(torch.zeros(1,2),s,2)
  self.assertEqual(v.sum(),0);torch.testing.assert_close(r,s['b_dec'][None])
 def test_normalization_inverse_and_pos0(self):
  h=torch.randn(1,9,7).bfloat16();m=torch.randn(1,7);std=torch.rand(1,7)+.1;x=(h[:,1:].float()-m)/std
  inv=x*std+m;torch.testing.assert_close(inv,h[:,1:].float(),atol=1e-6,rtol=1e-6)
  new=h.clone();new[:,1:]=inv.to(h.dtype);self.assertTrue(torch.equal(new,h))
if __name__=='__main__':unittest.main()
