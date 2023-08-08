import torch
from .REKD import REKD
from .hardnet_pytorch import HardNet
from .sosnet_pytorch import SOSNet32x32
from .hynet_pytorch import HyNet

from collections import OrderedDict

def load_detector(args, device):
    model1 = None
     
    if args.load_dir != '':
        args.group_size, args.dim_first, args.dim_second, args.dim_third = model_parsing(args)
        model1 = REKD(args, device)
        model1.load_state_dict(torch.load(args.load_dir))
        model1.export()
        model1.eval()
        model1.to(device) ## use GPU

    return model1
