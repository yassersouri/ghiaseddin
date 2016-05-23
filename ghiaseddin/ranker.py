import lasagne
import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
from pastalog import Log


class Ghiaseddin(object):
    _epsilon = 1.0e-7

    def __init__(self, extractor, dataset, train_batch_size=16, extractor_learning_rate=0.0001, ranker_learning_rate=0.0001, weight_decay=0.005):
        self.train_batch_size = train_batch_size
        self.extractor = extractor
        self.dataset = dataset
        self.weight_decay = weight_decay

        self.pastalog = Log('http://localhost:8100/', "e:%s-d:%s-bs:%d-lr:%f" % (self.extractor.NAME, self.dataset.NAME, self.train_batch_size, ranker_learning_rate))

        # TODO: check if converting these to shared variable actually improves performance.
        self.input_var = T.ftensor4('inputs')
        self.target_var = T.fvector('targets')

        self.extractor.set_input_var(self.input_var, batch_size=train_batch_size)
        self.extractor_layer = self.extractor.get_output_layer()

        self.extractor_learning_rate_shared_var = theano.shared(np.cast['float32'](extractor_learning_rate), name='extractor_learning_rate')
        self.ranker_learning_rate_shared_var = theano.shared(np.cast['float32'](ranker_learning_rate), name='ranker_learning_rate')

        self.extractor_params = lasagne.layers.get_all_params(self.extractor_layer, trainable=True)

        self.absolute_rank_estimate, self.ranker_params = self._absolute_rank_estimate(self.extractor_layer)
        self.reshaped_input = lasagne.layers.ReshapeLayer(self.absolute_rank_estimate, (-1, 2))

        # the posterior estimate layer is not trainable
        self.posterior_estimate = lasagne.layers.DenseLayer(self.reshaped_input, num_units=1, W=lasagne.init.np.array([[1], [-1]]), b=lasagne.init.Constant(val=0), nonlinearity=lasagne.nonlinearities.sigmoid)
        self.posterior_estimate.params[self.posterior_estimate.W].remove('trainable')
        self.posterior_estimate.params[self.posterior_estimate.b].remove('trainable')

        # the clipping is done to prevent the model from diverging as caused by binary XEnt
        self.predictions = T.clip(lasagne.layers.get_output(self.posterior_estimate).ravel(), self._epsilon, 1.0 - self._epsilon)

        self.xent_loss = lasagne.objectives.binary_crossentropy(self.predictions, self.target_var).mean()
        self.l2_penalty = lasagne.regularization.regularize_network_params(self.absolute_rank_estimate, lasagne.regularization.l2)
        self.loss = self.xent_loss + self.l2_penalty * weight_decay

        self.test_absolute_rank_estimate = lasagne.layers.get_output(self.absolute_rank_estimate, deterministic=True)

        self._create_theano_functions()

    def _create_theano_functions(self):
        """
        Will be creating theano functions for training and testing
        """
        self._feature_extractor_updates = lasagne.updates.rmsprop(self.loss, self.extractor_params, learning_rate=self.extractor_learning_rate_shared_var)
        self._ranker_updates = lasagne.updates.rmsprop(self.loss, self.ranker_params, learning_rate=self.ranker_learning_rate_shared_var)

        f = self._feature_extractor_updates.items()
        r = self._ranker_updates.items()
        f.extend(r)

        self._all_updates = OrderedDict(f)

        self.training_function = theano.function([self.input_var, self.target_var], [self.loss, self.xent_loss, self.l2_penalty], updates=self._all_updates)
        self.testing_function = theano.function([self.input_var], self.test_absolute_rank_estimate)

    def _absolute_rank_estimate(self, incoming):
        """
        An abstraction around the absolute rank estimate.
        Currently this is only a single dense layer with linear activation. This could easily be extended with more non-linearity.
        Should return the absolute rank estimate layer and all the parameters.
        """
        absolute_rank_estimate_layer = lasagne.layers.DenseLayer(incoming=incoming, num_units=1, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(val=0.0), nonlinearity=lasagne.nonlinearities.linear)
        # Take care if you want to use multiple layers here you have to return all the params of all the layers
        return absolute_rank_estimate_layer, absolute_rank_estimate_layer.get_params()

    def _train_1_batch(self, preprocessed_input, step):
        input_data, input_target, input_mask = preprocessed_input
        loss, xent_loss, l2_penalty = self.training_function(input_data, input_target)

        # log the losses
        self.pastalog.post('train_loss', value=float(loss), step=step)
        self.pastalog.post('train_xent', value=float(xent_loss), step=step)
        self.pastalog.post('train_l2pen', value=float(l2_penalty), step=step)

        return loss

    def train_one_epoch(self, add_step=0):
        train_generator = self.dataset.train_generator(batch_size=self.train_batch_size, shuffle=True, cut_tail=True)
        losses = []
        for i, b in enumerate(train_generator):
            preprocessed_input = self.extractor.preprocess(b)
            batch_loss = self._train_1_batch(preprocessed_input, add_step + i)
            losses.append(batch_loss)

        return losses
