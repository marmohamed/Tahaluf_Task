from thop import profile
import torch

def get_macs(model, device, args):
    input_1 = torch.randn(1, 3, args.width, args.height).to(device)
    macs, params = profile(model, inputs=(input_1, ))
    return macs, params
