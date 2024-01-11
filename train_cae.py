import numpy as np
import torch, math, warnings
import argparse, os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import dataloader, mean_and_var
from cae_models import CAE_finger

warnings.filterwarnings("ignore")

# # parse command line arguments
parser = argparse.ArgumentParser(description='Train CAE model')
parser.add_argument('--trainset', help='Train database folder', nargs='+', type=str)
parser.add_argument('--valset', help='Validation database folder', nargs='+', type=str)
parser.add_argument('--modelpath', help='Folder for model parameters', nargs='+', type=str)
parser.add_argument('--savefolder', help='Folder for saved figures', nargs='+', type=str)

args = parser.parse_args()
cae_train = args.trainset[0]
cae_val = args.valset[0]
path_to_model = args.modelpath[0]
path_to_saves = args.savefolder[0]

# database folders
#cae_train = './database/UTFVP_roi/train/'
#cae_val = './database/UTFVP_roi/val/'
# folders for saves
#path_to_model = './models/ROI/'
#path_to_saves = './figures/ROI/'

if not os.path.exists(path_to_model):
    os.mkdir(path_to_model)
if not os.path.exists(path_to_saves):
    os.mkdir(path_to_saves)
    os.mkdir(path_to_saves + '/input/')
    os.mkdir(path_to_saves + '/output/')


# generate grid image for gray scale images
def grid_image(images):
    # make grid image
    b_size = len(images)
    c, h, w = images[0].shape  # image size
    nCol = 8  # 16 images per row
    # number of images per column
    nRow = int(math.ceil(b_size / nCol))
    padding = 10
    h_grid = nRow * h + (nRow + 1) * padding
    w_grid = nCol * w + (nCol + 1) * padding
    grid_image = np.zeros((h_grid, w_grid))

    # fill grid image
    steph = padding + h
    stepw = padding + w
    rowTop = padding
    colLeft = padding
    for img in images:
        rowBottom = rowTop + h
        colRight = colLeft + w
        grid_image[rowTop:rowBottom, colLeft:colRight] = img
        colLeft = colLeft + stepw
        if colLeft >= w_grid:
            colLeft = padding
            rowTop = rowTop + steph
    return grid_image

def save_model(model, optimiser, epoch, path_to_model):
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optim_dict': optimiser.state_dict()}
    torch.save(state, path_to_model)

def train(train_loader, model, criterion, optimiser, device):
    # switch to train model
    model.train()
    running_loss = 0.0
    for idx, image in enumerate(train_loader):
        image = image.to(device)
        optimiser.zero_grad()
        output = model(image)
        loss = criterion(image, output)
        # compute grads and update
        loss.backward()
        optimiser.step()
        # sum all batch losses
        running_loss += loss.detach().item()
    avr_loss = running_loss / train_loader.__len__()
    return avr_loss

def validate(val_loader, model, criterion, optimiser, device):
    # randomly select a batch to generate grid image
    rand_batch = np.random.randint(val_loader.__len__(), size=1)[0]
    # rand_batch = np.random.randint(100, size=1)[0]
    # switch to train model
    model.eval()
    running_loss = 0.0

    input_images_val = []
    output_images_val = []
    for idx, image in enumerate(val_loader):
        image = image.to(device)
        optimiser.zero_grad()
        output = model(image)
        # save inout and output images if the batch is selected
        if idx == rand_batch:
            input_images_val.append(image.detach().cpu().numpy())
            output_images_val.append(output.detach().cpu().numpy())
        # Structural Similarity Index loss - reconstruction loss
        loss = criterion(image, output)
        # sum all batch losses
        running_loss += loss.detach().item()

    avr_loss = running_loss / val_loader.__len__()
    return avr_loss, input_images_val, output_images_val

def main():
    num_epochs_cae = 50
    batch_size = 64
    lr = 1e-4

    # train data for CAE
    mean, var = mean_and_var(cae_train)
    trainset = dataloader(cae_train, mode='train', mean=mean, std=var)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # validation set
    valset = dataloader(cae_val, mode='train', mean=mean, std=var)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Train set length: {}'.format(trainloader.__len__()))
    print('Validation set length: {}'.format(valloader.__len__()))

    # select cuda device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'No GPU found')

    # create cae network
    model = CAE_finger(num_layers=7, in_channels=16, disable_decoder=False).to(device)

    # optimiser
    optimiser_cae = optim.Adam(model.parameters(), lr=lr)

    # cae criterion/loss - L2 ( Mean Square Error )
    criterion_cae = nn.MSELoss()

    # train CAE
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs_cae):
        # define input and output images
        # train one epoch
        epoch_loss_train = train(trainloader, model, criterion_cae, optimiser_cae, device)
        epoch_loss_val, in_images, out_images = validate(valloader, model, criterion_cae, optimiser_cae, device)

        # every 10 epochs save figures and model parameters
        if epoch % 10 == 0 or epoch == num_epochs_cae - 1:
            # save validation images
            input_grid_val = grid_image(in_images[0])
            output_grid_val = grid_image(out_images[0])
            # save figures
            figin, axin = plt.subplots()
            axin.imshow(input_grid_val, cmap='gray')
            plt.title('Input images - Epoch {}'.format(epoch))
            plt.savefig(path_to_saves + '/input/input_grid_epoch{}.png'.format(epoch))
            plt.close(figin)

            figout, axout = plt.subplots()
            axout.imshow(output_grid_val, cmap='gray')
            plt.title('Output images - Epoch {}'.format(epoch))
            plt.savefig(path_to_saves + '/output/output_grid_epoch{}.png'.format(epoch))
            plt.close(figout)

            # save reconstructed images along with the feature space???
            modelfile = path_to_model + '/model_cae_epoch{}.pt'.format(epoch)
            save_model(model, optimiser_cae, epoch, modelfile)

        train_loss.append(epoch_loss_train)
        val_loss.append(epoch_loss_val)

        # display the epoch training loss
        print("epoch : {}/{}, train loss = {:.4f} validation loss(w) = {:.4f} ".format(epoch + 1, num_epochs_cae,
                                                                                    epoch_loss_train, epoch_loss_val))

    # plot rain loss of cae
    axEpoch = np.arange(1, num_epochs_cae + 1)
    figaeloss, axaeloss = plt.subplots()
    axaeloss.plot(axEpoch, train_loss, label='Train loss', color='b')
    axaeloss.plot(axEpoch, val_loss, label='Validation loss', color='r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.title('CAE Train/Val. Loss')
    plt.savefig(path_to_saves + '/train_val_loss.png')

if __name__ == '__main__':
    main()