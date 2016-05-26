import os
import boltons.fileutils
import numpy as np
import lasagne

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
lasagne.random.set_rng(np.random)

data_root = os.path.join(os.path.expanduser('~'), 'ghiaseddin')
model_root = os.path.join(data_root, 'models')

googlenet_weights = os.path.join(model_root, 'blvc_googlenet.pkl')
vgg16_weights = os.path.join(model_root, 'vgg16.pkl')
inceptionv3_weights = os.path.join(model_root, 'inception_v3.pkl')

result_models_root = os.path.join(model_root, 'results')
zappos_result_models_root = os.path.join(result_models_root, 'zappos')
lfw10_result_models_root = os.path.join(result_models_root, 'lfw10')

dataset_root = os.path.join(data_root, 'datasets')
zappos_root = os.path.join(dataset_root, 'Zappos50K')
lfw10_root = os.path.join(dataset_root, 'LFW10')

boltons.fileutils.mkdir_p(zappos_result_models_root)
boltons.fileutils.mkdir_p(zappos_root)
boltons.fileutils.mkdir_p(lfw10_result_models_root)
boltons.fileutils.mkdir_p(lfw10_root)
