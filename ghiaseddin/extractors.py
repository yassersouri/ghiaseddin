import lasagne
import utils

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer

import cPickle as pickle
import numpy as np


class Extractor(object):
    """
    The Feature Learning and Extractor Sub-Network
    This is the abstraction class for the feature extractor sub-network.
    You are able to:
        - create a custom feature extractor, by creating a class that extends this one.

    A custom extractor must have the following properties:
        - net
            this contains the network layers
        - input_var
            this can be set at a later time with `set_input_var` which is called by the model.
        - out_layer
            this layer is the output of the feature extraction part of the network. e.g. for VGG16 it is the fc7 layer
    """
    INPUT_LAYER_NAME = 'input'
    _input_height = 224
    _input_width = 224
    _input_raw_scale = 255
    _input_mean_to_subtract = [104, 117, 123]

    def __init__(self, weights):
        self.weights = weights

    @staticmethod
    def _get_weights_from_file(file_addr, weights_key):
        with open(file_addr, 'rb') as f:
            params = pickle.load(f)
            init_weights = params[weights_key]
            if 'mean value' in params.keys():
                self._input_mean_to_subtract = params['mean value']
        return init_weights

    def set_input_var(self, input_var, batch_size=None):
        # TODO: Are we sure this works? Because we are going to call this function after we have called the init.
        input_layer_size = self.net[self.INPUT_LAYER_NAME].shape
        input_layer_size[0] = batch_size
        self.net[self.INPUT_LAYER_NAME] = lasagne.layers.InputLayer(input_layer_size, input_var=input_var)

    def _general_image_preprocess(self, img):
        img = utils.resize_image(img, (self._input_height, self._input_height))

        img = img.transpose((2, 0, 1))
        img = img[::-1, ...] * self._input_raw_scale
        img[0, ...] -= self._input_mean_to_subtract[0]
        img[1, ...] -= self._input_mean_to_subtract[1]
        img[2, ...] -= self._input_mean_to_subtract[2]

        return img

    def output_for_image(self, image_addr):
        """
        This function gives you the output from the extractor for a single image.

        Please don't use this function in your code. What you should be doing is this
        ```
        out_f = extractor.get_output_function()
        y = out_f(x) # y is output while x is your input to the extractor

        ```

        This function is just here for historical reasons and for debugging.
        """
        img = utils.load_image(image_addr)
        img = self._general_image_preprocess(img)

        data = np.zeros((1, 3, self._input_height, self._input_width), dtype=np.float32)
        data[0, ...] = img

        inp = lasagne.utils.T.tensor4('inp')
        out = lasagne.layers.get_output(self.out_layer, inputs=inp, deterministic=True)

        return out.eval({inp: data}).flatten()

    def get_output_function(self):
        pass


class GoogLeNet(Extractor):
    """
    Todo: Add DropOut layers.
    """
    _input_height = 224
    _input_width = 224
    _input_raw_scale = 255
    _input_mean_to_subtract = [104, 117, 123]

    def __init__(self, weights):
        super(GoogLeNet, self).__init__(weights)

        def build_inception_module(name, input_layer, nfilters):
            # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
            net = {}
            net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
            net['pool_proj'] = ConvLayer(net['pool'], nfilters[0], 1, flip_filters=False)

            net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

            net['3x3_reduce'] = ConvLayer(input_layer, nfilters[2], 1, flip_filters=False)
            net['3x3'] = ConvLayer(net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

            net['5x5_reduce'] = ConvLayer(input_layer, nfilters[4], 1, flip_filters=False)
            net['5x5'] = ConvLayer(net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

            net['output'] = lasagne.layers.ConcatLayer([
                net['1x1'],
                net['3x3'],
                net['5x5'],
                net['pool_proj']])

            return {'{}/{}'.format(name, k): v for k, v in net.items()}

        net = {}
        net['input'] = lasagne.layers.InputLayer((None, 3, None, None))
        net['conv1/7x7_s2'] = ConvLayer(net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
        net['pool1/3x3_s2'] = PoolLayer(net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
        net['pool1/norm1'] = lasagne.layers.LocalResponseNormalization2DLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
        net['conv2/3x3_reduce'] = ConvLayer(net['pool1/norm1'], 64, 1, flip_filters=False)
        net['conv2/3x3'] = ConvLayer(net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
        net['conv2/norm2'] = lasagne.layers.LocalResponseNormalization2DLayer(net['conv2/3x3'], alpha=0.00002, k=1)
        net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

        net.update(build_inception_module('inception_3a', net['pool2/3x3_s2'], [32, 64, 96, 128, 16, 32]))
        net.update(build_inception_module('inception_3b', net['inception_3a/output'], [64, 128, 128, 192, 32, 96]))
        net['pool3/3x3_s2'] = PoolLayer(net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

        net.update(build_inception_module('inception_4a', net['pool3/3x3_s2'], [64, 192, 96, 208, 16, 48]))

        net.update(build_inception_module('inception_4b', net['inception_4a/output'], [64, 160, 112, 224, 24, 64]))
        net.update(build_inception_module('inception_4c', net['inception_4b/output'], [64, 128, 128, 256, 24, 64]))
        net.update(build_inception_module('inception_4d', net['inception_4c/output'], [64, 112, 144, 288, 32, 64]))
        net.update(build_inception_module('inception_4e', net['inception_4d/output'], [128, 256, 160, 320, 32, 128]))
        net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

        net.update(build_inception_module('inception_5a', net['pool4/3x3_s2'], [128, 256, 160, 320, 32, 128]))
        net.update(build_inception_module('inception_5b', net['inception_5a/output'], [128, 384, 192, 384, 48, 128]))

        net['pool5/7x7_s1'] = lasagne.layers.GlobalPoolLayer(net['inception_5b/output'])
        net['dropout5'] = lasagne.layers.DropoutLayer(net['pool5/7x7_s1'], p=0.4)

        self.net = net
        self.out_layer = net['dropout5']

        init_weights = self._get_weights_from_file(self.weights, 'param values')
        init_weights = init_weights[:-2]  # since we have chopped off the last two layers of the network (loss3/classifier and prob), we won't need those
        lasagne.layers.set_all_param_values(self.out_layer, init_weights)


class VGG16(Extractor):
    def __init__(self, weights):
        super(VGG16, self).__init__(weights)

        net = {}
        net['input'] = lasagne.layers.InputLayer((None, 3, 224, 224))
        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['fc6'] = lasagne.layers.DenseLayer(net['pool5'], num_units=4096)
        net['fc6_dropout'] = lasagne.layers.DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = lasagne.layers.DenseLayer(net['fc6_dropout'], num_units=4096)
        net['fc7_dropout'] = lasagne.layers.DropoutLayer(net['fc7'], p=0.5)

        self.net = net
        self.out_layer = net['fc7_dropout']

        init_weights = self._get_weights_from_file(self.weights, 'param values')
        init_weights = init_weights[:-2]  # since we have chopped off the last two layers of the network, we won't need those
        lasagne.layers.set_all_param_values(self.out_layer, init_weights)
