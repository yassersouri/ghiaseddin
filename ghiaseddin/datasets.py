import scipy.io
import os
import matplotlib.pylab as plt
import utils
import numpy as np
import itertools
import boltons.iterutils


class Dataset(object):
    """
    Base class for a dataset helper. Implements functionality while subclasses will focus on loading
    the data into the desired format.

    This helper needs the following properties to successfully perform the necessary actions:
        1. _ATT_NAMES: It is a 1 dimensional list or list-like object, containing string names for the attributes in the dataset.
        2. _image_adresses: It is a 1 dimensional list or list-like object, containing absolute image address for each image in the dataset.
        3. _train_pairs: It is a (n x 2) array where n in the number of training pairs and they contain index of the images as the image
        address is specified with that index in _image_adresses.
        4. _train_targets: It is a (n) shaped array where n in the number of training pairs and contains the target posterior for our method
        ($\in [0, 1]$).
        5. _test_pairs: Similar to _train_pairs but for testing pairs.
        6. _test_targets: Similar to _train_targets but for for testing pairs.

    Each dataset helper needs to implement its __init__ function which fills the above properties according to the way this data is stored
    on disk.
    """
    _ATT_NAMES = None
    _train_pairs = None
    _train_targets = None
    _test_pairs = None
    _test_targets = None
    _image_adresses = None

    def __init__(self, root, attribute_index):
        self.root = root
        self.attribute_index = attribute_index
        assert 0 <= attribute_index < len(self._ATT_NAMES)

    def get_name(self):
        return "%s-%d" % (self.__class__.__name__, self.attribute_index)

    def _show_image_path_target(self, img1_path, img2_path, target):
        if target > 0.5:
            print 'A is more %s than B (t: %2.2f)' % (self._ATT_NAMES[self.attribute_index], target)
        elif target < 0.5:
            print 'A is less %s than B (t: %2.2f)' % (self._ATT_NAMES[self.attribute_index], target)
        else:
            print 'A is the same as B in %s (t: %2.2f)' % (self._ATT_NAMES[self.attribute_index], target)

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(utils.load_image(img1_path))
        ax1.set_title('A')
        ax1.axis('off')
        ax2.imshow(utils.load_image(img2_path))
        ax2.set_title('B')
        ax2.axis('off')
        plt.show()

    def show_pair(self, pair_id, test=False):
        """
        Shows pairs of images in the dataset and their annotation (target) for the set attribute.
        """
        pair = self._test_pairs[pair_id, :] if test else self._train_pairs[pair_id, :]
        target = self._test_targets[pair_id] if test else self._train_targets[pair_id]

        img1_path = self._image_adresses[pair[0]]
        img2_path = self._image_adresses[pair[1]]

        self._show_image_path_target(img1_path, img2_path, target)

    def _iterate_pair_target(self, indices, values, targets):
        for i in indices:
            yield ((self._image_adresses[values[i, 0]], self._image_adresses[values[i, 1]]), targets[i])

    def train_generator(self, batch_size, shuffle=True, cut_tail=True):
        """
        Returns a generator which yields an array of size `batch_size` where each element of the array is a tuple of kind ((img1_path, img2_path), target) from the training set.
            e.g.: [((img1_path, img2_path), target), ((img1_path, img2_path), target), ...]

        `batch_size` must be an int.
        If `shuffle` is `True` then the items will be shuffled.
        If `cut_tail` is `True` then the last item from the generator might not have length equal to `batch_size`. It might have a length of less than `batch_size`.
        If `cut_tail` is `False` then all items from the generator will have the same length equal to `batch_size`. In order to achieve this some of the items from the dataset will not get generated.

        Example Usage:
        >>> for batch in dataset.train_generator(64):
        >>>     for (img1_path, img2_path), target in batch:
        >>>         # do something with the batch
        """
        indices = np.arange(len(self._train_targets))

        if shuffle:
            # shuffle the indices in-place
            np.random.shuffle(indices)

        to_return = boltons.iterutils.chunked_iter(self._iterate_pair_target(indices, self._train_pairs, self._train_targets), batch_size)

        if cut_tail:
            slice_size = int(len(self._train_targets) / batch_size)
            return itertools.islice(to_return, slice_size)
        else:
            return to_return

    def test_generator(self, batch_size):
        """
        Similar to `train_generator` but for the test set.

        `batch_size` must be an int.
        The last item from the generator might contain `None`. This means that the test data was not enough to fill the last batch.
        The user of the dataset must take care of these `None` values.
        """
        indices = np.arange(len(self._test_targets))

        return boltons.iterutils.chunked_iter(self._iterate_pair_target(indices, self._test_pairs, self._test_targets), batch_size, fill=None)


class Zappos50K1(Dataset):
    """The dataset helper class for Zappos50K-1, the coarse version of the dataset."""

    _ATT_NAMES = ['open', 'pointy', 'sporty', 'comfort']

    def __init__(self, root, attribute_index, split_index):
        super(Zappos50K1, self).__init__(root, attribute_index)
        self.split_index = split_index

        data_path = os.path.join(self.root, 'ut-zap50k-data')
        images_path = os.path.join(self.root, 'images')  # TODO: Why?
        imagepath_info = scipy.io.loadmat(os.path.join(data_path, 'image-path.mat'))['imagepath'].flatten()
        train_test_file = scipy.io.loadmat(os.path.join(data_path, 'train-test-splits.mat'))
        labels_file = scipy.io.loadmat(os.path.join(data_path, 'zappos-labels.mat'))

        train_info = train_test_file['trainIndexAll'].flatten()
        test_info = train_test_file['testIndexAll'].flatten()

        train_index = train_info[attribute_index].flatten()[split_index].flatten()
        test_index = test_info[attribute_index].flatten()[split_index].flatten()
        image_pairs_order = labels_file['mturkOrder'].flatten()[attribute_index].astype(int)

        # create placeholders
        self._train_pairs = np.zeros((len(train_index), 2), dtype=np.int)
        self._train_targets = np.zeros((len(train_index),), dtype=np.float32)
        self._test_pairs = np.zeros((len(test_index), 2), dtype=np.int)
        self._test_targets = np.zeros((len(test_index),), dtype=np.float32)

        # fill place holders
        self._image_adresses = [os.path.join(images_path, p[0]) for p in imagepath_info]
        self._fill_pair_target(train_index, image_pairs_order, self._train_pairs, self._train_targets)
        self._fill_pair_target(test_index, image_pairs_order, self._test_pairs, self._test_targets)

    def get_name(self):
        return "Zap1-%d-%d" % (self.attribute_index, self.split_index)

    def _fill_pair_target(self, indexes, pair_order, pairs, targets):
        for i, id in enumerate(indexes):
            pair_info = pair_order[id - 1]  # because of matlab indexing
            pairs[i, :] = pair_info[0:2] - 1
            if pair_info[3] == 1:
                targets[i] = 1.0
            elif pair_info[3] == 2:
                targets[i] = 0.0
            elif pair_info[3] == 3:
                targets[i] = 0.5
            else:
                raise Exception("invalid target")


class LFW10(Dataset):
    """The dataset helper class for LFW-10 dataset."""

    _ATT_NAMES = ['baldhead', 'darkhair', 'eyesopen', 'goodlooking', 'masculinelooking', 'mouthopen', 'smile', 'v_teeth', 'vforehead', 'young']

    def __init__(self, root, attribute_index):
        super(LFW10, self).__init__(root, attribute_index)
        self.root = os.path.join(self.root, 'LFW10')

        data_path = os.path.join(self.root, 'annotations')
        images_path = os.path.join(self.root, 'images')

        annotation_file_train = scipy.io.loadmat(os.path.join(data_path, '{}train.mat'.format(self._ATT_NAMES[attribute_index])))
        annotation_file_test = scipy.io.loadmat(os.path.join(data_path, '{}test.mat'.format(self._ATT_NAMES[attribute_index])))

        # the training set
        self._train_pairs = np.zeros((500, 2), dtype=np.int)
        self._train_targets = np.zeros((500,), dtype=np.float32)
        for i in xrange(500):
            self._train_pairs[i, 0] = int(annotation_file_train['images_compare'][i, 1][0][:-4]) - 1  # first to remove the '.jpg' part
            self._train_pairs[i, 1] = int(annotation_file_train['images_compare'][i, 2][0][:-4]) - 1  # , then to convert to index
            idx = np.argmax(annotation_file_train['attribute_strengths'][i, 1:])
            if idx == 0:
                self._train_targets[i] = 1.0  # image1 has more strength
            elif idx == 1:
                self._train_targets[i] = 0.0  # image1 has less strength
            else:
                self._train_targets[i] = 0.5  # two images have about the same strength

        self._test_pairs = np.zeros((500, 2), dtype=np.int)
        self._test_targets = np.zeros((500,), dtype=np.float32)
        for i in xrange(500):
            self._test_pairs[i, 0] = int(annotation_file_test['images_compare'][i, 1][0][:-4]) - 1  # first to remove the '.jpg' part
            self._test_pairs[i, 1] = int(annotation_file_test['images_compare'][i, 2][0][:-4]) - 1  # , then to convert to index
            idx = np.argmax(annotation_file_test['attribute_strengths'][i, 1:])
            if idx == 0:
                self._test_targets[i] = 1.0  # image1 has more strength
            elif idx == 1:
                self._test_targets[i] = 0.0  # image1 has less strength
            else:
                self._test_targets[i] = 0.5  # two images have about the same strength

        # fill place holders
        self._image_adresses = [os.path.join(images_path, '{}.jpg'.format(p + 1)) for p in xrange(2000)]


class PubFig(Dataset):
    """The dataset helper class for PubFig dataset."""

    _ATT_NAMES = ['Male', 'White', 'Young', 'Smiling', 'Chubby', 'VisibleForehead', 'BushyEyebrows', 'NarrowEyes', 'PointyNose', 'BigLips', 'RoundFace']

    def __init__(self, root, attribute_index):
        super(PubFig, self).__init__(root, attribute_index)

        data_path = self.root
        images_path = os.path.join(self.root, 'images')
        data_file = scipy.io.loadmat(os.path.join(data_path, 'data.mat'), appendmat=False)
        # self._ATT_NAMES = map(lambda x: x[0], data_file['attribute_names'][0])
        im_names = data_file['im_names'].squeeze()
        self._image_adresses = [os.path.join(images_path, im_names[i][0]) for i in xrange(len(im_names))]
        class_labels = data_file['class_labels'][:, 0]
        used_for_training = data_file['used_for_training'][:, 0]

        X = np.arange(len(im_names), dtype=np.int)
        y = np.zeros((len(im_names), len(self._ATT_NAMES)), dtype=np.int)
        for i in xrange(len(im_names)):
            y[i, :] = data_file['relative_ordering'][:, class_labels[i] - 1]
        Xtrain = X[np.where(used_for_training)]
        Xtest = X[np.where(used_for_training - 1)]
        ytrain = y[np.where(used_for_training)]
        ytest = y[np.where(used_for_training - 1)]

        idxs = list(itertools.combinations(range(len(Xtrain)), 2))
        self._train_pairs = np.zeros((len(idxs), 2), dtype=np.int)
        self._train_targets = np.zeros((len(idxs),), dtype=np.float32)
        for cnt, ij in enumerate(idxs):
            i, j = ij
            self._train_pairs[cnt][0] = Xtrain[i]
            self._train_pairs[cnt][1] = Xtrain[j]
            self._train_targets[cnt] = (ytrain[i, attribute_index] == ytrain[j, attribute_index]) * 0.5 +\
                                       (ytrain[i, attribute_index] > ytrain[j, attribute_index]) * 1.0

        idxs = list(itertools.combinations(range(len(Xtest)), 2))
        self._test_pairs = np.zeros((len(idxs), 2), dtype=np.int)
        self._test_targets = np.zeros((len(idxs),), dtype=np.float32)
        for cnt, ij in enumerate(idxs):
            i, j = ij
            self._test_pairs[cnt][0] = Xtest[i]
            self._test_pairs[cnt][1] = Xtest[j]
            self._test_targets[cnt] = (ytest[i, attribute_index] == ytest[j, attribute_index]) * 0.5 +\
                                      (ytest[i, attribute_index] > ytest[j, attribute_index]) * 1.0


class OSR(Dataset):
    """The dataset helper class for OSR dataset."""

    _ATT_NAMES = ['natural', 'open', 'perspective', 'size-large', 'diagonal-plane', 'depth-close']

    def __init__(self, root, attribute_index):
        super(OSR, self).__init__(root, attribute_index)

        data_path = self.root
        images_path = os.path.join(self.root, 'spatial_envelope_256x256_static_8outdoorcategories')
        data_file = scipy.io.loadmat(os.path.join(data_path, 'data.mat'), appendmat=False)
        # self._ATT_NAMES = map(lambda x: x[0], data_file['attribute_names'][0])
        im_names = data_file['im_names'].squeeze()
        self._image_adresses = [os.path.join(images_path, im_names[i][0]) for i in xrange(len(im_names))]
        class_labels = data_file['class_labels'][:, 0]
        used_for_training = data_file['used_for_training'][:, 0]

        X = np.arange(len(im_names), dtype=np.int)
        y = np.zeros((len(im_names), len(self._ATT_NAMES)), dtype=np.int)
        for i in xrange(len(im_names)):
            y[i, :] = data_file['relative_ordering'][:, class_labels[i] - 1]
        Xtrain = X[np.where(used_for_training)]
        Xtest = X[np.where(used_for_training - 1)]
        ytrain = y[np.where(used_for_training)]
        ytest = y[np.where(used_for_training - 1)]

        idxs = list(itertools.combinations(range(len(Xtrain)), 2))
        self._train_pairs = np.zeros((len(idxs), 2), dtype=np.int)
        self._train_targets = np.zeros((len(idxs),), dtype=np.float32)
        for cnt, ij in enumerate(idxs):
            i, j = ij
            self._train_pairs[cnt][0] = Xtrain[i]
            self._train_pairs[cnt][1] = Xtrain[j]
            self._train_targets[cnt] = (ytrain[i, attribute_index] == ytrain[j, attribute_index]) * 0.5 +\
                                       (ytrain[i, attribute_index] > ytrain[j, attribute_index]) * 1.0

        idxs = list(itertools.combinations(range(len(Xtest)), 2))
        self._test_pairs = np.zeros((len(idxs), 2), dtype=np.int)
        self._test_targets = np.zeros((len(idxs),), dtype=np.float32)
        for cnt, ij in enumerate(idxs):
            i, j = ij
            self._test_pairs[cnt][0] = Xtest[i]
            self._test_pairs[cnt][1] = Xtest[j]
            self._test_targets[cnt] = (ytest[i, attribute_index] == ytest[j, attribute_index]) * 0.5 +\
                                      (ytest[i, attribute_index] > ytest[j, attribute_index]) * 1.0
