import click
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__name__)))
from datetime import datetime as dt
import lasagne
import ghiaseddin


@click.command()
@click.argument('dataset', type=click.Choice(['zappos1', 'lfw']))
@click.argument('attribute', type=click.INT)
@click.argument('epochs', type=click.INT)
@click.argument('attribute_split', type=click.INT)
def main(dataset, attribute, epochs, attribute_split):
    si = attribute_split

    if dataset == 'zappos1':
        dataset = ghiaseddin.Zappos50K1(ghiaseddin.settings.zappos_root, attribute_index=attribute, split_index=si)
    elif dataset == 'lfw':
        dataset = ghiaseddin.LFW10(ghiaseddin.settings.lfw10_root, attribute_index=attribute)

    tic = dt.now()
    sys.stdout.write('===================AI: %d, A: %s, SI: %d===================\n' % (attribute, dataset._ATT_NAMES[attribute], si))
    sys.stdout.flush()

    googlenet = ghiaseddin.GoogLeNet(ghiaseddin.settings.googlenet_weights)

    model = ghiaseddin.Ghiaseddin(extractor=googlenet,
                                  dataset=dataset,
                                  weight_decay=1e-5,
                                  optimizer=lasagne.updates.rmsprop,
                                  ranker_learning_rate=1e-4,
                                  extractor_learning_rate=1e-5,
                                  ranker_nonlinearity=lasagne.nonlinearities.linear)

    for _ in range(epochs):
        model.train_one_epoch()
        sys.stdout.write("%2.4f\n" % model.eval_accuracy())
        sys.stdout.flush()

    model.save()
    toc = dt.now()
    print 'Took: %s' % (str(toc - tic))


if __name__ == '__main__':
    main()
