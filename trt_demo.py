# Canyu Xie
import torch
import datetime
import time
from tqdm import tqdm

import random
from PIL import Image
import numpy as np
import sys

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

from trt.utils import *
from trt import common

from models import ResNet18
from collections import OrderedDict

from data.cifar_parse import CIFAR10_PARSE

class ModelData(object):

    INPUT_SHAPE = (3, 32, 32)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32



def load_checkpoint(net, pretrained):
    print('loading pretrained model from %s' % pretrained)
    
    state_dict_total = torch.load(pretrained)['net']

    net_state_dict_rename = OrderedDict()

    net_dict = net.state_dict()

    state_dict = state_dict_total

    for k, v in state_dict.items():

        name = k[7:] # remove `module.`
        
        net_state_dict_rename[name] = v

    net_dict.update(net_state_dict_rename)
    net.load_state_dict(net_dict, strict=False)

    return net

def create_torch_model():
    model = ResNet18()
    model = load_checkpoint(model, pretrained='checkpoint/ckpt.pth')
    
    return model 

def main():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    onnx_model_file = 'output/onnx/resnet18.onnx'

    inference_times = 100 

    dataset_test = CIFAR10_PARSE('./datasets')
    # ==> pytorch test
    model_torch = create_torch_model()
    input_torch, _ = dataset_test.get_one_image_torch(idx=0)

    t_begin = time.time()
    model_torch.eval()
    model_torch.cuda()

    with torch.no_grad():
        for i in tqdm(range(inference_times)):
            outputs_torch = model_torch(input_torch.cuda())
    t_end = datetime.datetime.now()
    
    t_end = time.time()
    torch_time = (t_end - t_begin)/inference_times # get the pytorch inference time
    
    # ==> tensorRT test
    with build_engine_onnx(TRT_LOGGER, onnx_model_file) as engine:

        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Load a normalized test case into the host input page-locked buffer.
            dataset_test.get_one_image_trt(inputs[0].host, idx=0)
            
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            
            t_begin = time.time()
            for i in tqdm(range(inference_times)):
                trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t_end = time.time()
            trt_time = (t_end - t_begin)/inference_times # get the tensorRT inference time


    print("==> compare torch output and tensorrt output")
    
    np.testing.assert_almost_equal(outputs_torch[0].view(-1).cpu().data.numpy(), trt_outputs[0], decimal=3)

    print("==> Passed")

    print('==> Torch time: {:.5f} ms'.format(torch_time))
    print('==> TRT time: {:.5f} ms'.format(trt_time))

if __name__ =='__main__':
    main()
