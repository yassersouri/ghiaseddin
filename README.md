![Ghiaseddin](https://github.com/yassersouri/ghiaseddin/blob/master/images/ghiaseddin.png)

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

## Downloading files
If you want to perform training yourself, you need to download some files (initial weights files and dataset images).

### Downloading datasets

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

### Downloading initial weights (models pretrained on ILSVRC)

**GoogLeNet**

```bash
python /path/to/project/ghiaseddin/scripts/download-weights-googlenet.py
```

**VGG16**

```bash
python /path/to/project/ghiaseddin/scripts/download-weights-vgg16.py
```

## Running our experiments (reproducing our results)

We have used Titan Black, Titan X, and Titan 980 Ti GPUs to produce our results.

The random seed can be set at `ghiaseddin/settings.py`. We have used 0, 1 and 2 as our random seeds for Zappos50k2, LFW10, OSR and PubFig experiments. (Zappos50k1 already has 10 different splits of training data so we have only run the full experiment once with 0 as random seed)

To reproduce our results you can run the following scripts which will output the accuracies.

```bash
./run-zappos1.sh # for Zappos50k1 experiment
./run-zappos2.sh # for Zappos50k2 experiment
./run-lfw.sh # for LFW10 experiment
./run-osr.sh # for OSR experiment
./run-pubfig.sh # for PubFig experiment
```

## Doing your own experiments

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

Here are some example saliencies (Not all saliencies are easily interpretable as these):

**OSR - Natural**
![Natural](https://github.com/yassersouri/ghiaseddin/blob/master/images/natural-2.png)
![Natural](https://github.com/yassersouri/ghiaseddin/blob/master/images/natural-5.png)

**Zappos50k1 - Open**
![Open](https://github.com/yassersouri/ghiaseddin/blob/master/images/open-3.png)
![Open](https://github.com/yassersouri/ghiaseddin/blob/master/images/open-6.png)

**Zappos50k1 - Pointy**
![Pointy](https://github.com/yassersouri/ghiaseddin/blob/master/images/pointy-1.png)
![Pointy](https://github.com/yassersouri/ghiaseddin/blob/master/images/pointy-7.png)

**LFW10 - Bald Head**
![Bald Head](https://github.com/yassersouri/ghiaseddin/blob/master/images/baldhead-1.png)
![Bald Head](https://github.com/yassersouri/ghiaseddin/blob/master/images/baldhead-8.png)

**LFW10 - Good Looking**
![Good Looking](https://github.com/yassersouri/ghiaseddin/blob/master/images/goodlooking-2.png)
![Good Looking](https://github.com/yassersouri/ghiaseddin/blob/master/images/goodlooking-7.png)

## Reference

If you use this code in your research please consider citing our paper:

@inproceedings{souri2016dra,
  title={Deep Relative Attributes},
  author={Souri, Yaser and Noury, Erfan and Adeli, Ehsan},
  booktitle={ACCV},
  year={2016}
}


## Feedback

We are not experts in Theano and/or Lasagne or in Deep Learning. So please provide us with your feedback. If you find any issues inside the paper please contact Yasser Souri (yassersouri@gmail.com). If you have issues or feedback related to the code, please use the [Github issues](https://github.com/yassersouri/Ghiaseddin/issues) section and file a new issue.
