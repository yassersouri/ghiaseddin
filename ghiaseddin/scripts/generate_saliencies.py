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
@click.option('--dataset', type=click.Choice(['zappos1', 'lfw', 'osr', 'pubfig']), default='zappos1')
def main(dataset):
    results_folder = os.path.join(ghiaseddin.settings.result_models_root, 'saliencies', dataset)

    if dataset == 'zappos1':
        DS = ghiaseddin.Zappos50K1
    elif dataset == 'lfw':
        DS = ghiaseddin.LFW10
    elif dataset == 'osr':
        DS = ghiaseddin.OSR
    elif dataset == 'pubfig':
        DS = ghiaseddin.PubFig

    for AI in range(len(DS._ATT_NAMES)):
        ext = ghiaseddin.VGG16(weights=ghiaseddin.settings.vgg16_weights)
        if dataset == 'zappos1':
            dst = ghiaseddin.Zappos50K1(ghiaseddin.settings.zappos_root, attribute_index=AI, split_index=0)
        elif dataset == 'lfw':
            dst = ghiaseddin.LFW10(ghiaseddin.settings.lfw10_root, attribute_index=AI)
        elif dataset == 'osr':
            dst = ghiaseddin.OSR(ghiaseddin.settings.osr_root, attribute_index=AI)
        elif dataset == 'pubfig':
            dst = ghiaseddin.PubFig(ghiaseddin.settings.pubfig_root, attribute_index=AI)
        model = ghiaseddin.Ghiaseddin(ext, dst)
        try:
            model.load()
            
            boltons.fileutils.mkdir_p(results_folder)
            for i in range(10):
                fig = model.generate_saliency(size=1)
                fig.savefig(os.path.join(results_folder, '%s-%d.png') % (dst._ATT_NAMES[AI], i))

            sys.stdout.write("%s\n" % dst._ATT_NAMES[AI])
            sys.stdout.flush()
        except:
            sys.stdout.write('notfound %s\n' % dst._ATT_NAMES[AI])
            sys.stdout.flush()

if __name__ == '__main__':
    main()
