from __future__ import division
import theano
import numpy as np
import scipy as sc
from sklearn.manifold import TSNE
import lasagne
import sys
sys.path.append('../')
import ghiaseddin


def generate_feature_embedding(model, batch_size=128):
    inp = model.extractor.get_input_var()
    outp = lasagne.layers.get_output(model.extractor_layer, deterministic=True)
    outrank = lasagne.layers.get_output(model.absolute_rank_estimate, deterministic=True)
    emb_func = theano.function([inp], [outp, outrank])

    def iterate_minibatch(batch_size):
        for idx in xrange(int(np.ceil(len(model.dataset._image_addresses) / batch_size))):
            images = np.zeros((batch_size, 3, model.extractor._input_height, model.extractor._input_width))
            for i in xrange(batch_size):
                if idx * batch_size + i >= len(model.dataset._image_addresses):
                    images = images[:i, ...]
                    break
                pth = model.dataset._image_addresses[idx * batch_size + i]
                try:
                    im = ghiaseddin.utils.load_image(pth)
                except:
                    im = np.zeros((model.extractor._input_height, model.extractor._input_width, 3))
                images[i, ...] = model.extractor._general_image_preprocess(im)
            yield images

    deep_feats = np.zeros((len(model.dataset._image_addresses), 4096), dtype=np.float32)
    deep_ranks = np.zeros((len(model.dataset._image_addresses),), dtype=np.float32)
    idx = 0
    for images in iterate_minibatch(batch_size):
        emb, rank = emb_func(images.astype(np.float32))
        deep_feats[idx:idx + images.shape[0], :] = emb
        deep_ranks[idx:idx + images.shape[0]] = rank.squeeze()
        print (idx + 1), (idx + images.shape[0])
        idx += images.shape[0]
    return deep_feats, deep_ranks


if __name__ == '__main__':
    att_index = 1  # Pointy attribute
    dataset = ghiaseddin.datasets.Zappos50K1(root=ghiaseddin.settings.zappos_root, attribute_index=att_index, split_index=0)
    extractor = ghiaseddin.VGG16(weights=ghiaseddin.settings.vgg16_weights)
    model = ghiaseddin.Ghiaseddin(extractor=extractor, dataset=dataset)
    model.load()
    deep_feats, deep_ranks = generate_feature_embedding(model)
    rank_ordering = sc.stats.rankdata(deep_ranks)
    ranknet_transed = TSNE(random_state=12345).fit_transform(deep_feats)

    np.save('tsne_ranknet_zappos_{}.npy'.format(att_index), ranknet_transed)
    np.save('rankordering_zappos_{}.npy'.format(att_index), rank_ordering)
