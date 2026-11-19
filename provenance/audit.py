"""GPU reconstruction and behavioral-fidelity audit.

Math helpers require only PyTorch. Running main requires the optional model and
dataset dependencies, original checkpoints and explicit corpus access. Outputs
can contain corpus text provenance and token IDs, subject to separate rights review.
"""
import json, hashlib, time, pathlib, gc, ast
import torch
import torch.nn.functional as F
OUT=pathlib.Path('audit_outputs')
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
 return h.hexdigest()
def ce(logits,ids):
 # Every label is token t+1; position0 output remains unpatched in all arms.
 return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(ids.shape[0],-1)
def reconstruct(x,s,k):
 pre=(x-s['b_dec'])@s['W_enc']+s['b_enc']; v,ix=pre.topk(k,dim=-1);v=v.relu()
 # Sparse gather is mathematically the identical full dictionary TopK decoder.
 r=(s['W_dec'][ix]*v.unsqueeze(-1)).sum(-2)+s['b_dec']
 return r,v,ix
@torch.inference_mode()
def main():
 import argparse
 from transformers import AutoTokenizer, AutoModelForCausalLM
 from datasets import load_dataset
 from huggingface_hub import HfApi
 import config as C
 global OUT
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--checkpoint-dir',type=pathlib.Path,required=True)
 parser.add_argument('--output-dir',type=pathlib.Path,default=OUT)
 parser.add_argument('--dataset-revision',required=True)
 args=parser.parse_args()
 OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
 start=time.time();torch.set_num_threads(4);torch.manual_seed(20261116)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.set_float32_matmul_precision('highest')
 api=HfApi(); rev=args.dataset_revision
 files=sorted(p for p in api.list_repo_files(C.DATASET_NAME,repo_type='dataset',revision=rev) if p.startswith('sample/10BT/') and p.endswith('.parquet'))
 assert len(files)>1,files
 shard=files[-1];print('DATA SHARD',shard,flush=True)
 url=f'https://huggingface.co/datasets/{C.DATASET_NAME}/resolve/{rev}/{shard}'
 tok=AutoTokenizer.from_pretrained(C.MODEL_NAME,revision=C.MODEL_REVISION)
 corpus=load_dataset('parquet',data_files={'train':url},split='train',streaming=True).skip(1000)
 docs=[];seqs=[];packed=[];packed_sources=[]
 for row,doc in enumerate(corpus,start=1000):
  ids=tok.encode(doc['text'],add_special_tokens=False)
  if len(ids)<64: continue
  key=hashlib.sha256(doc['text'].encode()).hexdigest();di=len(docs)
  docs.append({'last_shard_row':row,'document_id':doc.get('id'),'url':doc.get('url'),'text_sha256':key,'full_token_count':len(ids),'first512_token_ids':ids[:512]})
  if len([s for s in seqs if s['kind']=='document'])<32:
   seqs.append({'kind':'document','doc_indices':[di],'token_ids':ids[:512]})
  # Preserve all tokens + EOS as train packer, retain origin of every token.
  packed.extend(ids+[tok.eos_token_id]);packed_sources.extend([di]*(len(ids)+1))
  while len(packed)>=512 and len([s for s in seqs if s['kind']=='packed'])<32:
   seqs.append({'kind':'packed','doc_indices':sorted(set(packed_sources[:512])),'token_ids':packed[:512]});packed=packed[512:];packed_sources=packed_sources[512:]
  if len(seqs)==64:break
 seqs.sort(key=lambda s: (s['kind'],s['doc_indices'][0]))
 # First 10 are a balanced sanity prefix; all 64 evaluated only if controls pass.
 seqs=seqs[:5]+seqs[32:37]+seqs[5:32]+seqs[37:]
 provenance={'dataset':C.DATASET_NAME,'revision':rev,'shard':shard,'shard_index':len(files)-1,'shard_count':len(files),'skip_rows_in_shard':1000,'model':C.MODEL_NAME,'model_revision':C.MODEL_REVISION,'training_overlap':'Not proven disjoint: training dataset revision/order and consumed document identities not recorded. Last shard deliberately avoids training/mining prefix; duplicate content elsewhere possible.','inputs':seqs,'documents':docs,'sources':{p:sha(pathlib.Path(__file__).with_name(p)) for p in ['train_sae.py','config.py']}}
 save('inputs.json',provenance)
 model=AutoModelForCausalLM.from_pretrained(C.MODEL_NAME,revision=C.MODEL_REVISION,dtype=torch.bfloat16).cuda().eval();model.requires_grad_(False)
 save('model_config.json',model.config.to_dict())
 rows=[];summary={};inventory=[]
 # Base logits kept on CPU fp32 to independently compare identity at each layer.
 bases=[]
 for q in seqs:
  ids=torch.tensor([q['token_ids']],device='cuda'); logits=model(ids,use_cache=False).logits
  bases.append({'logits':logits.cpu(),'ce':ce(logits,ids).cpu()})
 for layer in [10,20,30]:
  path=args.checkpoint_dir/f'layer{layer}_latest.pt';before=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False)
  cfg=ck['config'];assert cfg['K']==32 and cfg['EXCLUDE_LOCAL_POSITION_0'] and cfg['MODEL_REVISION']==C.MODEL_REVISION
  state={k:v.cuda() for k,v in ck['sae_state'].items()};mean=ck['mean'].cuda();std=ck['std'].cuda();age=ck['steps_since_fired'];step=ck['step']
  meta={'layer':layer,'checkpoint':path.name,'sha256_before':before,'step':step,'keys':list(ck),'tensor_dtypes':{k:str(v.dtype) for k,v in state.items()},'mean_shape':list(mean.shape),'std_shape':list(std.shape),'mean_min':mean.min().item(),'mean_max':mean.max().item(),'std_min':std.min().item(),'std_max':std.max().item(),'all_finite':all(torch.isfinite(v).all().item() for v in [mean,std,*state.values()]),'training_dead_count':int((age>=cfg['DEAD_THRESHOLD_STEPS']).sum()),'steps_since_fired_quantiles':torch.quantile(age.float(),torch.tensor([0.,.5,.9,.99,1.])).tolist(),'resampling_age':'Unavailable: steps_since_fired resets both on firing and resampling, not a resampling timestamp.'}
  assert meta['all_finite'] and meta['std_min']>0
  del ck;gc.collect()
  # Duplication: seeded 512 query directions versus all 81920, no full NxN.
  wd=state['W_dec'];norm=wd.norm(dim=-1);queries=torch.randperm(wd.shape[0],device='cuda')[:512]
  cos=(wd[queries]/norm[queries,None])@(wd/norm[:,None]).T;cos[torch.arange(len(queries),device='cuda'),queries]=-2
  best,idx=cos.max(-1);meta['decoder_norm_quantiles']=torch.quantile(norm,torch.tensor([0.,.5,.99,1.],device='cuda')).tolist();meta['duplication_sample']={'query_ids':queries.cpu().tolist(),'nearest_ids':idx.cpu().tolist(),'max_cosines':best.cpu().tolist(),'gt_0_9':int((best>.9).sum()),'gt_0_99':int((best>.99).sum())};del cos
  freq=torch.zeros(cfg['D_SAE'],dtype=torch.long,device='cuda');sx=torch.zeros(C.D_MODEL,device='cuda',dtype=torch.float64);sx2=sx.clone();rx=sx.clone();rx2=sx.clone();N=0
  for j,q in enumerate(seqs):
   ids=torch.tensor([q['token_ids']],device='cuda');mode={'value':'identity'};metrics={}
   def hook(mod,inp,out):
    nonlocal N,sx,sx2,rx,rx2
    h=out[0] if isinstance(out,tuple) else out;x=(h[:,1:].float()-mean)/std
    if mode['value']=='identity': replacement=h[:,1:].clone()
    elif mode['value']=='reconstruction':
     flat=x.reshape(-1,C.D_MODEL);r,v,ix=reconstruct(flat,state,cfg['K']);r=r.reshape_as(x)
     # Check algebra against literal dense training forward on first 2 tokens.
     if j==0:
      pre=(flat[:2]-state['b_dec'])@state['W_enc']+state['b_enc'];vv,ii=pre.topk(cfg['K'],dim=-1);dense=torch.zeros_like(pre).scatter_(-1,ii,vv.relu())@state['W_dec']+state['b_dec'];err=(dense-r.reshape(-1,C.D_MODEL)[:2]).abs().max().item();metrics['dense_sparse_max_abs']=err;assert err<1e-4,err
     active=v>0;freq.scatter_add_(0,ix[active],torch.ones_like(ix[active],dtype=torch.long))
     d=(flat-r.reshape_as(flat)).double();xd=flat.double();rd=d
     sx+=xd.sum(0);sx2+=(xd*xd).sum(0);rx+=rd.sum(0);rx2+=(rd*rd).sum(0);N+=len(flat)
     metrics.update({'normalized_sse':d.square().sum().item(),'normalized_train_mean_sse':xd.square().sum().item(),'elements':flat.numel(),'tokens':len(flat),'l0_mean':active.float().sum(-1).mean().item(),'original_sse':((r-x)*std).double().square().sum().item(),'original_train_mean_sse':(x*std).double().square().sum().item(),'inverse_roundtrip_max_abs':(((x*std+mean)-h[:,1:].float()).abs().max().item())})
     replacement=(r*std+mean).to(h.dtype)
    elif mode['value']=='mean': replacement=mean.expand_as(x).to(h.dtype)
    elif mode['value']=='zero': replacement=torch.zeros_like(h[:,1:])
    new=h.clone();new[:,1:]=replacement;assert torch.equal(new[:,0],h[:,0]);assert new.dtype==h.dtype
    return (new,)+out[1:] if isinstance(out,tuple) else new
   handle=model.model.layers[layer].register_forward_hook(hook)
   losses={'intact':bases[j]['ce'].mean().item()};losses_after={'intact':bases[j]['ce'][:,1:].mean().item()}
   for arm in ['identity','reconstruction','mean','zero']:
    mode['value']=arm;logits=model(ids,use_cache=False).logits;per=ce(logits,ids)
    losses[arm]=per.mean().item();losses_after[arm]=per[:,1:].mean().item()
    if arm=='identity':
     err=(logits.cpu().float()-bases[j]['logits'].float()).abs().max().item();assert err==0,err;metrics['identity_logits_max_abs']=err
    assert torch.isfinite(per).all()
   handle.remove();rows.append({'layer':layer,'sequence':j,'kind':q['kind'],'length':ids.shape[1],'ce_tokens':ids.shape[1]-1,'loss':losses,'loss_positions1plus':losses_after,**metrics})
   with (OUT/'per_sequence.jsonl').open('a') as f:f.write(json.dumps(rows[-1])+'\n')
   if j==9 or j==63: print(json.dumps({'layer':layer,'completed':j+1,'elapsed_s':time.time()-start,'last':rows[-1]}),flush=True)
   if j==9:save(f'layer{layer}_sanity10.json',rows[-10:])
  meta['sha256_after']=sha(path);assert before==meta['sha256_after'];inventory.append(meta)
  torch.save({'frequency':freq.cpu(),'query_feature_ids':queries.cpu(),'nearest_ids':idx.cpu(),'nearest_cosines':best.cpu()},OUT/f'layer{layer}_health.pt')
  layerrows=[r for r in rows if r['layer']==layer];groups={}
  for group in ['all','document','packed']:
   rr=[r for r in layerrows if group=='all' or r['kind']==group];n=sum(r['ce_tokens'] for r in rr);loss={arm:sum(r['loss'][arm]*r['ce_tokens'] for r in rr)/n for arm in rr[0]['loss']};den=loss['mean']-loss['intact'];sse=sum(r['normalized_sse'] for r in rr);base=sum(r['normalized_train_mean_sse'] for r in rr)
   groups[group]={'sequences':len(rr),'ce_tokens':n,'loss':loss,'mean_minus_intact':den,'loss_recovered_fraction':(loss['mean']-loss['reconstruction'])/den if den>1e-5 else None,'reconstruction_ce_increase':loss['reconstruction']-loss['intact'],'normalized_mse_training_definition':sse/sum(r['elements'] for r in rr),'normalized_error_vs_training_mean':sse/base,'explained_variance_vs_training_mean':1-sse/base,'original_error_vs_training_mean':sum(r['original_sse'] for r in rr)/sum(r['original_train_mean_sse'] for r in rr)}
  centered=(sx2-sx.square()/N).sum().item();rescenter=(rx2-rx.square()/N).sum().item();total_sse=rx2.sum().item()
  summary[str(layer)]={'groups':groups,'pooled_heldout_centered_FVU':total_sse/centered,'pooled_variance_explained':1-rescenter/centered,'audit_tokens':N,'features_fired':int((freq>0).sum()),'features_not_observed':int((freq==0).sum()),'frequency_count_quantiles':torch.quantile(freq.float(),torch.tensor([0.,.5,.9,.99,1.],device='cuda')).tolist(),'training_dead_count':meta['training_dead_count']}
  save('summary.json',summary);save('checkpoint_inventory.json',inventory)
  del state,mean,std,wd;gc.collect();torch.cuda.empty_cache()
 save('run.json',{'elapsed_s':time.time()-start,'start_unix':start,'end_unix':time.time(),'gpu_peak_allocated':torch.cuda.max_memory_allocated(),'torch':torch.__version__,'transformers':__import__('transformers').__version__,'script_sha256':sha(__file__)})
 print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
