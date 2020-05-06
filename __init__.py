import torch
from .CSPDarknet53_SPP_PANet import CSPDarknet53_SPP_PANet


def _load(weights_file):
    model = CSPDarknet53_SPP_PANet()
    model.load_state_dict(torch.load(weights_file))
    return model
