from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import mxnet as mx


ctx = mx.cpu(0)
image_size = (112, 112)
prefix = "../models/resnet-50"
epoch = 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix,
                                                       epoch)

all_layers = sym.get_internals()
sym = all_layers['relu1_output']
dellist = []
for k,v in arg_params.iteritems():
  if k.startswith('fc1'):
    dellist.append(k)
for d in dellist:
  del arg_params[d]
mx.model.save_checkpoint(prefix+"s", 0, sym, arg_params, aux_params)

digraph = mx.viz.plot_network(sym, shape={'data':(1,3,256,256)},
node_attrs={"fixedsize":"false"})
digraph.view()
