import torch
from torch.autograd import Variable
import torch.onnx as torch_onnx
import onnx
from models import ResNet18

from collections import OrderedDict


def load_checkpoint(net, pretrained):
    print('loading pretrained model from %s' % pretrained)
    
    state_dict_total = torch.load(pretrained)['net']

    net_state_dict_rename = OrderedDict()

    net_dict = net.state_dict()

    state_dict = state_dict_total
    # print(state_dict.keys())
    # pdb.set_trace()
    for k, v in state_dict.items():

        name = k[7:] # remove `module.`
        
        net_state_dict_rename[name] = v

    net_dict.update(net_state_dict_rename)
    net.load_state_dict(net_dict, strict=False)

    return net

def main():
    input_shape = (3, 32, 32)
    model_onnx_path = "output/onnx/resnet18.onnx"
    dummy_input = Variable(torch.randn(1, *input_shape))
    
    # get model and load checkpoint
    model = ResNet18()
    model = load_checkpoint(model, pretrained='checkpoint/ckpt.pth')

    model.train(False)
    
    # export onnx
    inputs = ['input.1']
    outputs = ['ouput.1']
    # dynamic_axes = {'input.1': {0: 'batch'}, 'ouput.1':{0:'batch'}}
    out = torch.onnx.export(model, dummy_input, model_onnx_path, input_names=inputs, output_names=outputs)


if __name__=='__main__':
    main()
