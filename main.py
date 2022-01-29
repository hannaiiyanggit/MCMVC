# =====================
# View Labels Are Important: A Multifacet Complementarity Study of Deep Multi-view Clustering
# =====================
# Contributor: Chuanxing Geng and Aiyang Han (equal contribution)
# Date: Jan, 2022
# }
# =====================

import os
import argparse
import collections
import itertools
import torch
import gc

from MCMVC import MCMVC
from get_mask import get_mask
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config

dataset = {
    0: "Caltech101-20",
    1: "Scene_15",
    2: "LandUse_21",
    3: "NoisyMNIST"
}

loss = {
    0: "mse",
    1: "infoNCE_loss"
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--instance_loss', type=int, default='0', help='instance-level loss id')
parser.add_argument('--missing_rate', type=float, default='0.0', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]
if args.missing_rate==0.0:
    completeness=0
else:
    completeness=1



def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset, args.instance_loss*2+completeness)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    config['instance_loss'] = loss[args.instance_loss]
    config['training']['missing_rate'] = args.missing_rate
    logger = get_logger()

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    # Load data
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]
    del X_list
    gc.collect()

    accumulated_metrics = collections.defaultdict(list)

    for data_seed in range(1, args.test_time + 1):
        # Get the Mask
        np.random.seed(data_seed)
        mask = get_mask(2, x1_train_raw.shape[0], config['training']['missing_rate'])

        # mask the data
        x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]
        x2_train = torch.from_numpy(x2_train).float().to(device)

        mask = torch.from_numpy(mask).long().to(device)

        # Set random seeds
        if config['training']['missing_rate'] == 0:
            seed = data_seed
        else:
            seed = config['training']['seed']
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Build the model
        model = MCMVC(config)
        optimizer = torch.optim.Adam(
            itertools.chain(model.autoencoder1.parameters(), model.autoencoder2.parameters(),
                            model.img2txt.parameters(), model.txt2img.parameters(),
                            model.classifier1.parameters()),
            lr=config['training']['lr'])
        model.to_device(device)

        # Print the models
        logger.info(model.autoencoder1)
        logger.info(model.classifier1)
        logger.info(optimizer)

        # Training
        acc, nmi, ari = model.train(config, logger, x1_train, x2_train, Y_list,
                                        mask, optimizer, device, data_seed)
        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['nmi'].append(nmi)
        accumulated_metrics['ari'].append(ari)

    logger.info('--------------------Training over--------------------')
    cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'])

    logger.handlers = []


if __name__ == '__main__':
    main()

