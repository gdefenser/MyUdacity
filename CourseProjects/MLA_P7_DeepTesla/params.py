import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf

#### CONST
## Initialize Constant
flags = tf.app.flags
FLAGS = flags.FLAGS

## Nvida's camera format
flags.DEFINE_integer('img_h', 64, 'The image height.')
flags.DEFINE_integer('img_w', 64, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')

## Fix random seed for reproducibility
np.random.seed(42)

## Path
parent_path='E:/Lovesha/Udacity/Projects/MyUdacity.git/trunk/CourseProjects/MLA_P6_proposal/deep_tesla'
data_dir = parent_path+'/epochs'
out_dir = parent_path+'/output'
model_dir = parent_path+'/models'