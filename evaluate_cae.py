import numpy as np
import matplotlib.pyplot as plt
import torch, os, argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import dataloader, mean_and_var
from cae_models import CAE_finger
from collections import OrderedDict
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# # parse command line arguments
parser = argparse.ArgumentParser(description='CAE evaluation')
parser.add_argument('--matedset', help='Mated database folder', nargs='+', type=str)
parser.add_argument('--nonmateset', help='Non-mated database folder', nargs='+', type=str)
parser.add_argument('--weights', help='Model weights', nargs='+', type=str)
parser.add_argument('--savefolder', help='Folder for saved figures and files', nargs='+', type=str)

args = parser.parse_args()
matedset = args.matedset[0]
nonmatedset = args.nonmatedset[0]
weights = args.weights[0]
savefolder = args.savefolder()

#matedset = './database/UTFVP_roi_eval/mated/'
#nonmatedset = './database/UTFVP_roi_eval/nonmated/'
#weights = './models/ROI/model_cae_epoch29.pt'
#savefolder = './figures/ROI/eval/'

# load cae model
def load_model(model, path, device):
    model_parameters = torch.load(path, map_location=device)
    model_weights = model_parameters['state_dict']
    state_dict_remove_module = OrderedDict()
    for k, v in model_weights.items():
        state_dict_remove_module[k] = v
    model.load_state_dict(state_dict_remove_module)
    return model.to(device)

def compare(dataloader, model, device):
    similarities = []

    for pair in dataloader:
        reference = pair[0].to(device)
        probe = pair[1].to(device)

        # image encodings
        _, LR = model(reference)
        _, LP = model(probe)

        # pair similarity
        sim = F.cosine_similarity(LR, LP).detach().item()
        similarities.append(sim)

    return similarities

def evaluate(genuine, imposter):
    labels = [1] * len(genuine) + [0] * len(imposter)
    scores = np.append(genuine, imposter)

    # evaluation
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auc = roc_auc_score(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return [fpr, tpr], auc, eer

def compute_similarity_histograms(genuine, imposter):
    scores = np.append(genuine, imposter)
    labels = [1] * len(genuine) + [0] * len(imposter)
    stepSize = .02
    axisRange = np.append(np.arange(0.0, 1.0, stepSize), 1.0)
    nRange = axisRange.size
    axisNGenuine = np.zeros(nRange)
    axisNImposter = np.zeros(nRange)
    for score, label in zip(scores, labels):
        rangeIndex = 0
        while (score > axisRange[rangeIndex]) and (rangeIndex < nRange):
            rangeIndex += 1
        if label == 1:
            axisNGenuine[rangeIndex] += 1.
        else:
            axisNImposter[rangeIndex] += 1.
    nGenuine = np.count_nonzero(labels)
    nImposter = len(labels) - nGenuine
    axisNImposter /= nImposter / 100
    axisNGenuine /= nGenuine / 100

    return axisNGenuine, axisNImposter

def main():
    trainset = './database/utfvp_128-256_roiShift_cl2.0_train/'
    mean, var = mean_and_var(trainset)

    # read from lmdb database
    matedData = dataloader(matedset, mode='eval', mean=mean, std=var)
    matedLoader = DataLoader(matedData, batch_size=1, num_workers=0, shuffle=False)
    nonMatedData = dataloader(nonmatedset, mode='eval', mean=mean, std=var)
    nonMatedLoader = DataLoader(nonMatedData, batch_size=1, num_workers=0, shuffle=False)

    # define the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CAE_finger(num_layers=7, in_channels=16, disable_decoder=True)

    model = load_model(model, weights, device)
    model.eval()

    # mated pair similarities
    matedSim = compare(matedLoader, model, device)

    # nonmated pair similarities
    nonMatedSim = compare(nonMatedLoader, model, device)

    # evaluate
    [fpr, tpr], auc, eer = evaluate(matedSim, nonMatedSim)

    # similarity histogram
    matedHist, nonMatedHist = compute_similarity_histograms(matedSim, nonMatedSim)

    print('EER: {:.4f}\tAUC: {:.4f}'.format(eer, auc))

    # folder for saved figures and files

    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    # save results
    np.savetxt(savefolder + '/auc_eer.csv', [auc, eer], delimiter=',')
    np.savetxt(savefolder + '/fpr.csv', fpr, delimiter=',')
    np.savetxt(savefolder + '/tpr.csv', tpr, delimiter=',')
    np.savetxt(savefolder + '/mated_sim.csv', matedSim, delimiter=',')
    np.savetxt(savefolder + '/non_mated_sim.csv', nonMatedSim, delimiter=',')
    np.savetxt(savefolder + '/mated_histogram.csv', matedHist, delimiter=',')
    np.savetxt(savefolder + 'non_mated_histogram.csv', nonMatedHist, delimiter=',')

    # plot roc curve
    figROC, axROC = plt.subplots()
    axROC.plot(fpr, tpr)
    axROC.set_xlabel('FPR')
    axROC.set_ylabel('TPR')
    axROC.set_title('ROC Curve')
    plt.savefig(savefolder + '/roc_curve.png')
    plt.close()

    # plot similarity histogram
    xAxis = np.append(np.arange(0.0, 1.0, .02), 1.0)
    figHist, axHist = plt.subplots()
    axHist.plot(xAxis, matedHist, label='Mated', c='indigo')
    axHist.plot(xAxis, nonMatedHist, label='Non-Mated', c='darkorange')
    axHist.set_xlabel('Cosine Similarity')
    axHist.set_ylabel('Count(%)')
    axHist.legend(loc='best')
    axHist.set_title('Similarity histogram')
    plt.savefig(savefolder + '/similarity_histogram.png')
    plt.close()

if __name__ == '__main__':
    main()