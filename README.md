![Ghiaseddin](https://github.com/yassersouri/ghiaseddin/blob/master/ghiaseddin.png)
# Ghiaseddin - قیاس الدین

This repo contains the code for the paper "Deep Relative Attributes" by Yaser Souri, Erfan Noury, Ehsan Adeli Mosabbeb.

## The paper

**Deep Relative Attributes** by Yaser Souri ([@yassersouri](https://github.com/yassersouri)), Erfan Noury ([@erfannoury](https://github.com/erfannoury)), Ehsan Adeli Mosabbeb ([@eadeli](https://github.com/eadeli)). ACCV 2016.

The paper on arXiv: [arxiv:1512.04103](http://arxiv.org/abs/1512.04103)

## The name

The name is "Ghiaseddin" which is written as "قیاس الدین" in Persian/Arabic. It is pronouned as "Ghiyāth ad-Dīn". Ghias or "قیاس" is the Persia/Arabic word that refers to the act of comparing two things (which is what we actually do in relative attributes).
Furthermore Ghiaseddin has a relation to [Ghiyāth al-Dīn Jamshīd al-Kāshī](https://en.wikipedia.org/wiki/Jamsh%C4%ABd_al-K%C4%81sh%C4%AB) "غیاث الدین جمشید کاشانی", where "Ghiaseddin" is pronounced similar to the first name of "Jamshīd al-Kāshī" but written with different letters in Persian/Arabic ("قیاس الدین" vs "غیاث الدین").

## Dependencies

The code is written in Python 2.7 and uses the [Lasagne](https://github.com/Lasagne/Lasagne) deep learning framework which is based on the amazing [Theano](https://github.com/Theano/Theano). These two are the main dependencies of the project. Besides these you will be needing CUDA 7 and cuDNN 4. It might work without CUDA or with lower versions but I have not tested it.

To visualize the training procedure I have used [pastalog](https://github.com/rewonc/pastalog) which you will have to install.

For a complete list of dependencies and their versions see `requirements.txt`.

## Running the experiments

### Downloading the necessary files

If you want to perform training yourself, you need to download some files (initial weights files and dataset images).

#### Downloading datasets

**Zappos50K**

```bash
python /path/to/project/ghiaseddin/scripts/download-dataset-zappos.py
```

**LFW10**

```bash
python /path/to/project/ghiaseddin/scripts/download-dataset-lfw10.py
```

**OSR and PubFig**

```bash
python /path/to/project/ghiaseddin/scripts/download-dataset-osr_pubfig.py
```

#### Downloading initial weights (models pretrained on ILSVRC)

**GoogLeNet**

```bash
python /path/to/project/ghiaseddin/scripts/download-weights-googlenet.py
```

**VGG16**

```bash
python /path/to/project/ghiaseddin/scripts/download-weights-vgg16.py
```

### Training a new model

First start the pastalog server.

```bash
/path/to/project/ghiaseddin/scripts/start_pastalog.sh
```

Then you can use ghiaseddin to train:

```python
import sys
sys.path.append('/path/to/ghiaseddin/')
import ghiassedin

zappos = ghiaseddin.Zappos50K1(ghiaseddin.settings.zappos_root, attribute_index=0, split_index=0)
googlenet = ghiaseddin.GoogeLeNet(ghiaseddin.settings.googlenet_ilsvrc_weights)
model = ghiaseddin.Ghiaseddin(extractor=googlenet, dataset=zappos) # possibility to add other options

# train the model for 10 epochs
losses = []
for i in range(10):
    loss = model.train_one_epoch()
    losses.append(loss)

# or like this
losses = model.train_n_epoch(10) # here losses is a list of size 10

# save the trained model
model.save('/path/to/model.pkl')
```

### Calculating accuracy of a model

```python
# calculates the relative attribute prediction accuracy
print model.eval_accuracy()
```

### Visualizing saliency

```python
# randomly generates saliency maps for 10 samples of the testing set
fig = model.generate_saliency(size=10)
# or you can specify which pairs
fig = model.generate_saliency([10, 20, 30, 40])
# and you can easily save the figure
fig.savefig('/path/to/file/saliency.png')
```

## Feedback

We are not experts in Theano and/or Lasagne or in Deep Learning. So please provide us with your feedback. If you find any issues inside the paper please contact Yasser Souri (yassersouri@gmail.com). If you have issues or feedback related to the code, please use the [Github issues](https://github.com/yassersouri/Ghiaseddin/issues) section and file a new issue.
