import os, cv2

class FVImage:
    pass

def read_images_UT(pathData, numSample=20, mode='train'):
    # read images from pathData
    # Parameters:
    #   pathData - path to finger vein images
    #       > string
    #   numSample - number of subjects to be read from
    #       the dataset
    #       > integer: default - 20
    #
    # Created on: 1-7-2019

    # finger image struct list
    fingersPerPerson = []
    # get the folders in pathData
    imageFolders = os.listdir(pathData)
    if mode == 'train':
        imageFolders = imageFolders[0:numSample]
    elif mode == 'eval':
        imageFolders = imageFolders[-numSample:]

    # Replace len(imageFolders) with numSamples
    for fldr in range(len(imageFolders)):
        # get the images in the folder of a subject
        imagesInFolder = os.listdir(pathData + imageFolders[fldr])
        imagesPerFinger = []
        fingerId = 1
        for img in range(len(imagesInFolder)-1):
            # image path
            imgPath = pathData + '/' + imageFolders[fldr] + '/' + imagesInFolder[img]
            fvImage = FVImage()
            fvImage.image = cv2.imread(imgPath,0)
            fvImage.p = fldr + 1
            fvImage.f = int(imagesInFolder[img].split('_')[1])
            fvImage.i = int(imagesInFolder[img].split('_')[2])
            # store the image belonging to the same finger together
            imagesPerFinger.append(fvImage)
            fingerId += 1
            if fingerId > 4:
                fingerId = 1
                # store all the finger images belonging to the same finger together
                fingersPerPerson.append(imagesPerFinger)
                imagesPerFinger = []

    return fingersPerPerson