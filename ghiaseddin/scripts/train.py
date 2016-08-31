import click
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__name__)))
from datetime import datetime as dt
import lasagne
import ghiaseddin
import boltons.fileutils
import numpy as np


@click.command()
@click.option('--dataset', type=click.Choice(['zappos1', 'lfw']), default='zappos1')
@click.option('--augmentation', type=click.BOOL, default=False)
@click.option('--attribute', type=click.INT, default=0)
@click.option('--epochs', type=click.INT, default=10)
@click.option('--attribute_split', type=click.INT, default=0)
@click.option('--do_log', type=click.BOOL, default=True, envvar='DO_LOG')
def main(dataset, augmentation, attribute, epochs, attribute_split, do_log):
    print do_log; exit;

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
                                  ranker_nonlinearity=lasagne.nonlinearities.linear,
                                  do_log=do_log)
    # saliency stuff
    test_pair_ids = np.random.choice(range(len(dataset._test_targets)), size=10)
    saliency_folder_path = os.path.join(ghiaseddin.settings.result_models_root, "saliency|%s" % model.NAME)
    boltons.fileutils.mkdir_p(saliency_folder_path)

    matrixes = []
    matrixes.append(model.estimates_predictions_corrects_on_test())
    accuracies = []
    for _ in range(epochs):
        model.train_one_epoch()
        acc = model.eval_accuracy() * 100
        accuracies.append(acc)
        sys.stdout.write("%2.4f\n" % acc)
        sys.stdout.flush()
        matrixes.append(model.estimates_predictions_corrects_on_test())

        # save saliency figure
        fig = model.generate_saliency(test_pair_ids)
        fig.savefig(os.path.join(saliency_folder_path, 'saliency-%d.png' % model.log_step))

        # save conv1 filters
        model.conv1_filters()

    model.save()

    # save missclassified
    model.generate_misclassified()

    # save corrects pairs during training
    folder_path = os.path.join(ghiaseddin.settings.result_models_root, "matrixes|%s" % model.NAME)
    boltons.fileutils.mkdir_p(folder_path)
    fig = ghiaseddin.utils.show_training_matrixes([ghiaseddin.utils.convert_estimates_on_test_to_matrix(c) for e, p, c in matrixes], 'Corrects')
    fig.savefig(os.path.join(folder_path, 'corrects-%d.png' % model.log_step))

    # Save raw accuracy values to file
    boltons.fileutils.mkdir_p(ghiaseddin.settings.result_models_root)
    with(open(os.path.join(ghiaseddin.settings.result_models_root, 'acc|%s' % model._model_name_with_iter()), 'w')) as f:
        f.write('\n'.join(["%2.4f" % a for a in accuracies]))
        f.write('\n')

    toc = dt.now()
    print 'Took: %s' % (str(toc - tic))


if __name__ == '__main__':
    main()
