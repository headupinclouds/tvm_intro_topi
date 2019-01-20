"""
Introduction to TOPI
====================
**Author**: `Ehsan M. Kermani <https://github.com/ehsanmok>`_

This is an introductory tutorial to TVM Operator Inventory (TOPI).
TOPI provides numpy-style generic operations and schedules with higher abstractions than TVM.
In this tutorial, we will see how TOPI can save us from writing boilerplates code in TVM.
"""
from __future__ import absolute_import, print_function

import tvm
import topi
import numpy as np
import argparse

target_table = [
    'llvm',
    'llvm -target=aarch64-linux-android',
    'cuda',
    'opengl',
    'opencl',
    'vulkan',
    'metal'
];

description="""
TVM from_mxnet tutorial with cross platform modifications\n
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=description
)

parser.add_argument(
    '--target',
    choices=[x for x in target_table],
    help="Target device for inference step",
)

parser.add_argument(
    '--target-host',
    default=None,
    help="Host device (cross platform usage)",
)

args = parser.parse_args()

target = args.target
target_host = None if (args.target_host == "None") else args.target_host

print("target", target)
print("target_host", target_host)

# detect android cross compilation
is_android = True if ('android' in (target + str(target_host))) else False

from topi.util import get_const_tuple

######################################################################
# Fusing convolutions
# -------------------
# We can fuse :code:`topi.nn.conv2d` and :code:`topi.nn.relu` together.
#
# .. note::
#
#    TOPI functions are all generic functions. They have different implementations
#    for different backends to optimize for performance.
#    For each backend, it is necessary to call them under a target scope for both
#    compute declaration and schedule. TVM will choose the right function to call with
#    the target information.

if target=='None':
    ctx = tvm.cpu(0)
elif target=='llvm -target=aarch64-linux-android':
    ctx = tvm.cpu(0)
elif target=='llvm':
    ctx = tvm.cpu(0)
elif target=='cuda':
    ctx = tvm.gpu(0)
elif target=='opengl':
    ctx = tvm.opengl(0)
elif target=='opencl':
    ctx = tvm.cl(0)
elif target=='vulkan':
    ctx = tvm.vulkan(0)
elif target=='metal':
    ctx = tvm.metal(0)
else:
    raise ValueError('No supported context type for ' % target)

data = tvm.placeholder((1, 3, 224, 224), name='data')
kern = tvm.placeholder((10, 3, 5, 5), name='kern')

dtype = data.dtype

data_np = np.ones(shape=get_const_tuple(data.shape)).astype(dtype)
kern_np = np.ones(shape=get_const_tuple(kern.shape)).astype(dtype) * (1.0/75.0)

with tvm.target.create(target):
    conv = topi.nn.conv2d(data, kern, strides=1, padding=2)
    outp = topi.nn.relu(conv)
    sconv = topi.generic.nn.schedule_conv2d_nchw([outp])

print(tvm.lower(sconv, [data, kern, outp], simple_mode=True))

#func = tvm.build(sconv, [data, kern, outp], target=target, target_host=target_host, name="intro_topi")

func = tvm.build(sconv, [data, kern, outp], target=target, target_host=target_host, name="intro_topi")

if is_android:
    print("build for android")
    
    func.export_library('intro_topi.so', tvm.contrib.ndk.create_shared, options=[
        "-g",
        "-shared",
        "-fPIC",
        "-nostdlib++"
    ])

    # TODO: we could enable the android rpc server for device testing in python
    # skip inference step for cross compilation for now
    exit()
else:
    func.export_library("intro_topi.so")

outp_shape = get_const_tuple(outp.shape)
data_ = tvm.nd.array(data_np, ctx)
kern_ = tvm.nd.array(kern_np, ctx)
outp_ = tvm.nd.array(np.zeros(outp_shape, dtype=dtype), ctx)

func(data_, kern_, outp_)

print("outp", outp_)
print("out_shape ", outp_shape)
print("done")
