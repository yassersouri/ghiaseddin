import lasagne
import utils

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers.dnn import Pool2DDNNLayer as Pool2DLayer
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
        return init_weights

    def set_input_var(self, input_var, batch_size=None):
        input_layer_shape = list(self.net[self.INPUT_LAYER_NAME].shape)
        input_layer_shape[0] = batch_size
        self.net[self.INPUT_LAYER_NAME].input_var = input_var
        self.net[self.INPUT_LAYER_NAME].shape = tuple(input_layer_shape)

    def _general_image_preprocess(self, img):
        img = utils.resize_image(img, (self._input_height, self._input_height))

        img = img.transpose((2, 0, 1))
        img = self._input_raw_scale * img[::-1, ...]
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

    def get_output_layer(self):
        return self.out_layer

    def preprocess(self, batch):
        batch_size = len(batch)
        images = np.zeros((batch_size * 2, 3, self._input_height, self._input_width), dtype=np.float32)
        annotations = np.zeros((batch_size), dtype=np.float32)
        mask = np.ones((batch_size), dtype=np.int8)

        for i, batch_item in enumerate(batch):
            if batch_item is None:
                mask[i] = 0
                continue

            (img1_path, img2_path), target = batch_item
            images[2 * i, ...] = self._general_image_preprocess(utils.load_image(img1_path))
            images[2 * i + 1, ...] = self._general_image_preprocess(utils.load_image(img2_path))
            annotations[i] = target

        return images, annotations, mask


class GoogLeNet(Extractor):
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
        net['input'] = lasagne.layers.InputLayer((None, 3, 224, 224))
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
    _input_mean_to_subtract = [104, 117, 123]
    _input_raw_scale = 255
    _input_height = 224
    _input_width = 224

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


class InceptionV3(Extractor):
    _input_height = 299
    _input_width = 299
    _input_mean_to_subtract = [0, 0, 0]

    def _general_image_preprocess(self, img):
        img = utils.resize_image(img, (self._input_height, self._input_height))

        img = img.transpose((2, 0, 1))
        img = 2 * img[::-1, ...] - 1  # change image from (0, 1) to (-1, 1)
        img[0, ...] -= self._input_mean_to_subtract[0]
        img[1, ...] -= self._input_mean_to_subtract[1]
        img[2, ...] -= self._input_mean_to_subtract[2]

        return img

    def __init__(self, weights):
        super(InceptionV3, self).__init__(weights)

        def bn_conv(input_layer, **kwargs):
            l = ConvLayer(input_layer, **kwargs)
            l = lasagne.layers.BatchNormLayer(l, epsilon=0.001)
            return l


        def inceptionA(input_layer, nfilt):
            # Corresponds to a modified version of figure 5 in the paper
            l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

            l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
            l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

            l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
            l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
            l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

            l4 = Pool2DLayer(
                input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
            l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

            return lasagne.layers.ConcatLayer([l1, l2, l3, l4])


        def inceptionB(input_layer, nfilt):
            # Corresponds to a modified version of figure 10 in the paper
            l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

            l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
            l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
            l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

            l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

            return lasagne.layers.ConcatLayer([l1, l2, l3])


        def inceptionC(input_layer, nfilt):
            # Corresponds to figure 6 in the paper
            l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

            l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
            l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
            l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

            l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
            l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
            l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
            l3 = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
            l3 = bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

            l4 = Pool2DLayer(
                input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
            l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

            return lasagne.layers.ConcatLayer([l1, l2, l3, l4])


        def inceptionD(input_layer, nfilt):
            # Corresponds to a modified version of figure 10 in the paper
            l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
            l1 = bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

            l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
            l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
            l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
            l2 = bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

            l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

            return lasagne.layers.ConcatLayer([l1, l2, l3])


        def inceptionE(input_layer, nfilt, pool_mode):
            # Corresponds to figure 7 in the paper
            l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

            l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
            l2a = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
            l2b = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

            l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
            l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
            l3a = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
            l3b = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

            l4 = Pool2DLayer(
                input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

            l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

            return lasagne.layers.ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])

        net = {}
        net['input'] = lasagne.layers.InputLayer((None, 3, 299, 299))
        net['conv'] = bn_conv(net['input'],
                            num_filters=32, filter_size=3, stride=2)
        net['conv_1'] = bn_conv(net['conv'], num_filters=32, filter_size=3)
        net['conv_2'] = bn_conv(net['conv_1'],
                                num_filters=64, filter_size=3, pad=1)
        net['pool'] = Pool2DLayer(net['conv_2'], pool_size=3, stride=2, mode='max')

        net['conv_3'] = bn_conv(net['pool'], num_filters=80, filter_size=1)

        net['conv_4'] = bn_conv(net['conv_3'], num_filters=192, filter_size=3)

        net['pool_1'] = Pool2DLayer(net['conv_4'],
                                    pool_size=3, stride=2, mode='max')
        net['mixed/join'] = inceptionA(
            net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
        net['mixed_1/join'] = inceptionA(
            net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

        net['mixed_2/join'] = inceptionA(
            net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

        net['mixed_3/join'] = inceptionB(
            net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

        net['mixed_4/join'] = inceptionC(
            net['mixed_3/join'],
            nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

        net['mixed_5/join'] = inceptionC(
            net['mixed_4/join'],
            nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

        net['mixed_6/join'] = inceptionC(
            net['mixed_5/join'],
            nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

        net['mixed_7/join'] = inceptionC(
            net['mixed_6/join'],
            nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

        net['mixed_8/join'] = inceptionD(
            net['mixed_7/join'],
            nfilt=((192, 320), (192, 192, 192, 192)))

        net['mixed_9/join'] = inceptionE(
            net['mixed_8/join'],
            nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
            pool_mode='average_exc_pad')

        net['mixed_10/join'] = inceptionE(
            net['mixed_9/join'],
            nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
            pool_mode='max')

        net['pool3'] = lasagne.layers.GlobalPoolLayer(net['mixed_10/join'])

        self.net = net
        self.out_layer = net['pool3']

        init_weights = self._get_weights_from_file(self.weights, 'param values')
        init_weights = init_weights[:-2]  # since we have chopped off the last two layers of the network (loss3/classifier and prob), we won't need those
        lasagne.layers.set_all_param_values(self.out_layer, init_weights)
