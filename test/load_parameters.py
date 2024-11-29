import torch

# state_dict = torch.load('../out/vlp-MMT50-LoRA-241014-1100/checkpoint.pth', map_location='cpu')
state_dict = torch.load('/nas3/ziihun/out/mbart50_lora_re/best_checkpoint.pth', map_location='cpu')

for name, param in state_dict['model'].items():
    print(name)
    # print(param)