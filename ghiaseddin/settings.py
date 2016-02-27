import os
import boltons.fileutils


data_root = os.path.join(os.path.expanduser('~'), 'ghiaseddin')
model_root = os.path.join(data_root, 'models')

googlenet_weights = os.path.join(model_root, 'blvc_googlenet.pkl')
vgg16_weights = os.path.join(model_root, 'vgg16.pkl')

result_models_root = os.path.join(model_root, 'results')
zappos_result_models_root = os.path.join(result_models_root, 'zappos')

dataset_root = os.path.join(data_root, 'datasets')
zappos_root = os.path.join(dataset_root, 'Zappos50K')

boltons.fileutils.mkdir_p(zappos_result_models_root)
boltons.fileutils.mkdir_p(zappos_root)
