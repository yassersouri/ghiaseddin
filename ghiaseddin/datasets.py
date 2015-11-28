import scipy.io
import os
import matplotlib.pylab as plt
import utils


class DatasetHelper(object):
    def __init__(self, root, attribute_index):
        self.root = root
        self.attribute_index = attribute_index


class Zappos50K1(DatasetHelper):
    """The dataset helper class for Zappos50K-1, the coarse version of the dataset."""

    ATT_NAMES = ['open', 'pointy', 'sporty', 'comfort']

    def __init__(self, root, attribute_index, split_index):
        super(Zappos50K1, self).__init__(root, attribute_index)
        self.split_index = split_index
        data_path = os.path.join(self.root, 'data')
        self.images_path = os.path.join(self.root, 'images')
        self.imagepath_info = scipy.io.loadmat(os.path.join(data_path, 'image-path.mat'))['imagepath'].flatten()
        train_test_file = scipy.io.loadmat(os.path.join(data_path, 'train-test-splits.mat'))
        labels_file = scipy.io.loadmat(os.path.join(data_path, 'zappos-labels.mat'))

        train_info = train_test_file['trainIndexAll'].flatten()
        test_info = train_test_file['testIndexAll'].flatten()

        self.train_index = train_info[attribute_index].flatten()[split_index].flatten()
        self.test_index = test_info[attribute_index].flatten()[split_index].flatten()
        self.image_pairs_order = labels_file['mturkOrder'].flatten()[attribute_index].astype(int)

    def vis_pair(self, pair_id, test=False):
        if test:
            pair_info = self.image_pairs_order[self.test_index[pair_id] - 1]
        else:
            pair_info = self.image_pairs_order[self.train_index[pair_id] - 1]

        pair_label = pair_info[3]
        if pair_label == 1:
            print 'A is more %s than B' % self.ATT_NAMES[self.attribute_index]
        else:
            print 'A is less %s than B' % self.ATT_NAMES[self.attribute_index]
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        img1_path = os.path.join(self.images_path, self.imagepath_info[pair_info[0] - 1][0])
        img2_path = os.path.join(self.images_path, self.imagepath_info[pair_info[1] - 1][0])
        ax1.imshow(utils.load_image(img1_path))
        ax1.set_title('A')
        ax1.axis('off')
        ax2.imshow(utils.load_image(img2_path))
        ax2.set_title('B')
        ax2.axis('off')
        plt.show()


class Zappos50K2(DatasetHelper):
    """The dataset helper class for Zappos50K-2, the fine-grained version of the dataset."""

    ATT_NAMES = ['open', 'pointy', 'sporty', 'comfort']

    def __init__(self, root, attribute_index):
        super(Zappos50K2, self).__init__(root, attribute_index)
