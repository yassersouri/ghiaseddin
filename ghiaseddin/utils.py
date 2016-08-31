import skimage
import skimage.io
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
import matplotlib.pylab as plt
import keras_image_preprocessing

# The following two function are borrowed from Caffe
# https://github.com/BVLC/caffe/blob/32dc03f14c36d1df46f37a7d13ad528e52c6f786/python/caffe/io.py#L278-L337


def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.
    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).
    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)


def convert_estimates_on_test_to_matrix(predictions, height=10):
    predictions = np.reshape(predictions, (-1, 1)).T
    predictions = np.resize(predictions, (height, predictions.shape[1]))
    return predictions


def show_training_matrixes(estimates, title):
    length = len(estimates)
    axes = []
    fig = plt.figure(figsize=(100, 2 * length))
    fig.suptitle(title, fontsize=30, verticalalignment='top')
    for i in range(length):
        axes.append(fig.add_subplot(length, 1, i + 1))

    with plt.rc_context({'image.cmap': 'gray', 'image.interpolation': 'nearest'}):
        for i in range(length):
            axes[i].matshow(estimates[i])
            axes[i].axis('off')
    return fig


def _random_fliprl(img):
    if np.random.rand() > 0.5:
        return np.fliplr(img)
    else:
        return img


def _random_rotate(img):
    return keras_image_preprocessing.random_rotation(img, 20, row_index=0, col_index=1, channel_index=2)


def _random_zoom(img):
    return keras_image_preprocessing.random_zoom(img, (0.65, 0.6), row_index=0, col_index=1, channel_index=2)


def random_augmentation(img):
    img = _random_fliprl(img)
    img = _random_zoom(img)
    img = _random_rotate(img)
    return img
