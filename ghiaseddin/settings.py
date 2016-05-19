import os
import boltons.fileutils


data_root = os.path.join(os.path.expanduser('~'), 'ghiaseddin')
model_root = os.path.join(data_root, 'models')

googlenet_weights = os.path.join(model_root, 'blvc_googlenet.pkl')
vgg16_weights = os.path.join(model_root, 'vgg16.pkl')
inceptionv3_weights = os.path.join(model_root, 'inception_v3.pkl')

result_models_root = os.path.join(model_root, 'results')
zappos_result_models_root = os.path.join(result_models_root, 'zappos')
lfw10_result_models_root = os.path.join(result_models_root, 'lfw10')
osr_result_models_root = os.path.join(result_models_root, 'osr')
pubfig_result_models_root = os.path.join(result_models_root, 'pubfig')

dataset_root = os.path.join(data_root, 'datasets')
zappos_root = os.path.join(dataset_root, 'Zappos50K')
lfw10_root = os.path.join(dataset_root, 'LFW10')
osr_pubfig_root = os.path.join(dataset_root, 'OSR-PubFig')
osr_root = os.path.join(osr_pubfig_root, 'relative_attributes', 'osr')
pubfig_root = os.path.join(osr_pubfig_root, 'relative_attributes', 'pubfig')

boltons.fileutils.mkdir_p(zappos_result_models_root)
boltons.fileutils.mkdir_p(zappos_root)
boltons.fileutils.mkdir_p(lfw10_result_models_root)
boltons.fileutils.mkdir_p(lfw10_root)
boltons.fileutils.mkdir_p(osr_result_models_root)
boltons.fileutils.mkdir_p(pubfig_result_models_root)
boltons.fileutils.mkdir_p(osr_pubfig_root)
