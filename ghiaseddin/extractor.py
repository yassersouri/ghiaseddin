import lasagne

try:
    import lasagne.layers.dnn.Pool2DDNNLayer as Pool2DLayer
    import lasagne.layers.dnn.Conv2DDNNLayer as Conv2DLayer
except:
    import lasagne.layers.Pool2DLayer as Pool2DLayer
    import lasagne.layers.Conv2DLayer as Conv2DLayer


class Extractor(object):
    """
    The Feature Learning and Extractor Sub-Network
    This is the abstraction class for the feature extractor sub-network.
    You are able to:
        - create a custom feature extractor, by creating a class that extends this one.

    A custom extractor must have the following properties:
        - net
            this contains the network layers
        - layers
            this contains the name of the layers, not exactly in the order which they appear in the network
        - input_var
            this can be set at a later time with `set_input_var` which is called by the model.
    """
    INPUT_LAYER_NAME = 'input'

    def __init__(self, weights=None):
        self.weights = weights

    def set_input_var(self, input_var, batch_size=None):
        # TODO: Are we sure this works? Because we are going to call this function after we have called the init.
        input_layer_size = self.net[self.INPUT_LAYER_NAME].shape
        input_layer_size[0] = batch_size
        self.net[self.INPUT_LAYER_NAME] = lasagne.layers.InputLayer(input_layer_size, input_var=input_var)


class GoogLeNet(Extractor):
    def __init__(self, weights):
        super(Extractor, self).__init__(weights)


class VGG16(Extractor):
    def __init__(self, weights):
        super(Extractor, self).__init__(weights)

        net = {}
        net['input'] = lasagne.layers.InputLayer((None, 3, 224, 224), input_var=None)
        net['conv1_1'] = Conv2DLayer(net['input'], 64, 3, pad=1)
        net['conv1_2'] = Conv2DLayer(net['conv1_1'], 64, 3, pad=1)
        net['pool1'] = Pool2DLayer(net['conv1_2'], 2)
        net['conv2_1'] = Conv2DLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_2'] = Conv2DLayer(net['conv2_1'], 128, 3, pad=1)
        net['pool2'] = Pool2DLayer(net['conv2_2'], 2)
        net['conv3_1'] = Conv2DLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_2'] = Conv2DLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_3'] = Conv2DLayer(net['conv3_2'], 256, 3, pad=1)
        net['pool3'] = Pool2DLayer(net['conv3_3'], 2)
        net['conv4_1'] = Conv2DLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_2'] = Conv2DLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_3'] = Conv2DLayer(net['conv4_2'], 512, 3, pad=1)
        net['pool4'] = Pool2DLayer(net['conv4_3'], 2)
        net['conv5_1'] = Conv2DLayer(net['pool4'], 512, 3, pad=1)
        net['conv5_2'] = Conv2DLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_3'] = Conv2DLayer(net['conv5_2'], 512, 3, pad=1)
        net['pool5'] = Pool2DLayer(net['conv5_3'], 2)
        net['fc6'] = lasagne.layers.DenseLayer(net['pool5'], num_units=4096)
        net['fc7'] = lasagne.layers.DenseLayer(net['fc6'], num_units=4096)
        # net['fc8'] = lasagne.layers.DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
        # net['prob'] = lasagne.layers.NonlinearityLayer(net['fc8'], lasagne.nonlinearities.softmax)
        # net_output = net['prob']
        self.net = net
        self.layers = net.keys()
