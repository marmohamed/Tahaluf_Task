from thop import profile
import torch

def get_macs(model, args):
    input_1 = torch.randn(1, 3, args.width, args.height)
    macs, params = profile(model, inputs=(input_1, ))
    return macs, params
