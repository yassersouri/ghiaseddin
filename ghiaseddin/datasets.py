import scipy.io
import os
import matplotlib.pylab as plt
import utils
import numpy as np


class DatasetHelper(object):
    """
    Base class for a dataset helper. Implements functionality while subclasses will focus on loading
    the data into the desired format.

    This helper needs the following properties to successfully perform the necessary actions:
        1. ATT_NAMES: It is a 1 dimensional list or list-like object, containing string names for the attributes in the dataset.
        2. image_adresses: It is a 1 dimentional list or list-like object, containing absolute image address for each image in the dataset.
        3. train_pairs: It is a (n x 2) array where n in the number of training pairs and they contain index of the images as the image
        address is specified with that index in image_adresses.
        4. train_targets: It is a (n) array where n in the number of training pairs and contains the target posterior for our method
        ($\in [0, 1]$).
        5. test_pairs: Similar to train_pairs but for testing pairs.
        6. test_targets: Similar to train_targets but for for testing pairs.

    Each dataset helper needs to implement its __init__ function which fills the above properties according to the way this data is stored
    on disk.
    """
    ATT_NAMES = None
    train_pairs = None
    train_targets = None
    test_pairs = None
    test_targets = None
    image_adresses = None

    def __init__(self, root, attribute_index):
        self.root = root
        self.attribute_index = attribute_index
        assert 0 <= attribute_index < len(self.ATT_NAMES)

    def show_pair(self, pair_id, test=False):
        """
        Shows pairs of images in the dataset and their annotation (target) for the set attribute.
        """
        pair = self.test_pairs[pair_id, :] if test else self.train_pairs[pair_id, :]
        target = self.test_targets[pair_id] if test else self.train_targets[pair_id]

        img1_path = self.image_adresses[pair[0]]
        img2_path = self.image_adresses[pair[1]]

        if target > 0.5:
            print 'A is more %s than B (t: %2.2f)' % (self.ATT_NAMES[self.attribute_index], target)
        elif target < 0.5:
            print 'A is less %s than B (t: %2.2f)' % (self.ATT_NAMES[self.attribute_index], target)
        else:
            print 'A is the same as B in %s (t: %2.2f)' % (self.ATT_NAMES[self.attribute_index], target)

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


class Zappos50K1(DatasetHelper):
    """The dataset helper class for Zappos50K-1, the coarse version of the dataset."""

    ATT_NAMES = ['open', 'pointy', 'sporty', 'comfort']

    def __init__(self, root, attribute_index, split_index):
        super(Zappos50K1, self).__init__(root, attribute_index)
        self.split_index = split_index

        data_path = os.path.join(self.root, 'data')
        images_path = os.path.join(self.root, 'images')
        imagepath_info = scipy.io.loadmat(os.path.join(data_path, 'image-path.mat'))['imagepath'].flatten()
        train_test_file = scipy.io.loadmat(os.path.join(data_path, 'train-test-splits.mat'))
        labels_file = scipy.io.loadmat(os.path.join(data_path, 'zappos-labels.mat'))

        train_info = train_test_file['trainIndexAll'].flatten()
        test_info = train_test_file['testIndexAll'].flatten()

        train_index = train_info[attribute_index].flatten()[split_index].flatten()
        test_index = test_info[attribute_index].flatten()[split_index].flatten()
        image_pairs_order = labels_file['mturkOrder'].flatten()[attribute_index].astype(int)

        # create placeholders
        self.train_pairs = np.zeros((len(train_index), 2), dtype=np.int)
        self.train_targets = np.zeros((len(train_index),), dtype=np.float32)
        self.test_pairs = np.zeros((len(test_index), 2), dtype=np.int)
        self.test_targets = np.zeros((len(test_index),), dtype=np.float32)

        # fill place holders
        self.image_adresses = [os.path.join(images_path, p[0]) for p in imagepath_info]
        self._fil_pair_target(train_index, image_pairs_order, self.train_pairs, self.train_targets)
        self._fil_pair_target(test_index, image_pairs_order, self.test_pairs, self.test_targets)

    def _fil_pair_target(self, indexes, pair_order, pairs, targets):
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
