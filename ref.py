import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
mp = "/Users/mac/bbverifier/vibethinker-bbtriage-v2"
tok = AutoTokenizer.from_pretrained(mp)
model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float32)
model.eval()
prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
ids = tok(prompt, return_tensors='pt').input_ids
ref = {"ids": ids.tolist()[0]}
emb = model.model.embed_tokens(ids)
ref["embed_last_8"] = [round(x,5) for x in emb[0,-1,:8].tolist()]
cap = {}
model.model.layers[0].register_forward_hook(lambda m,i,o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
with torch.no_grad():
    logits = model(ids).logits
ref["layer0_last_8"] = [round(x,5) for x in cap['h'][0,-1,:8].tolist()]
last = logits[0,-1]
top = torch.topk(last,5)
ref["argmax"] = int(last.argmax())
ref["argmax_tok"] = tok.decode([int(last.argmax())])
ref["top5_ids"] = top.indices.tolist()
ref["top5_vals"] = [round(v,3) for v in top.values.tolist()]
json.dump(ref, open("ref.json","w"), indent=1)
print(json.dumps(ref, indent=1))

gen = model.generate(ids, max_new_tokens=16, do_sample=False)
g = gen[0, ids.shape[1]:].tolist()
ref["gen_ids"] = g
ref["gen_text"] = tok.decode(g)
json.dump(ref, open("ref.json","w"), indent=1)
print("GEN_IDS", g)
print("GEN_TEXT", repr(tok.decode(g)))

sd = model.state_dict()
for k in ['model.layers.0.input_layernorm.weight','model.layers.0.self_attn.q_proj.weight','model.layers.0.self_attn.q_proj.bias','model.layers.0.mlp.down_proj.weight']:
    ref['W_'+k] = [round(x,5) for x in sd[k].flatten()[:8].tolist()]
json.dump(ref, open("ref.json","w"), indent=1)
print("WEIGHTS_DUMPED")
