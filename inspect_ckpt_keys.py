import torch

path = r'd:\Cuselin\Project\satmae_pp-main\checkpoint_ViT-L_pretrain_fmow_sentinel.pth'
ckpt = torch.load(path, map_location='cpu')
state = ckpt.get('model', ckpt.get('state_dict', ckpt))

print(f"Top-level keys: {list(ckpt.keys())[:5]}")
print(f"State dict size: {len(state)}")

sample = [k for k in state.keys()][:20]
print("Sample keys:")
for k in sample:
    print(" ", k)

has_decoder = any(k.startswith('decoder.') for k in state.keys())
print("Has MAE decoder keys?:", has_decoder)