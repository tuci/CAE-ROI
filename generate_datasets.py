import cv2
from process_image_set import compute_properties_image_set, extract_roi
from read_images import read_images_UT
from augmentation import crop_image, flip_image
from lmdb_database import make_lmdb_utfvp
from generate_pairs import generate_genuine_pairs, select_object_images
from register_images import register_images

def generate_lmdb_utfvp_train(datapath, dbpath, nCaeSub=20):
    # datapath - path to raw image
    # dbpath - base path for database
    # network type of network, cae or cnn
    # nCaeSub - number of subjects used for cae training
    #           default is 20

    # read images
    images = read_images_UT(datapath)
    # cae subjects
    start = 0
    end = nCaeSub * 6
    cae_subjects = images[start:end]

    # compute image properties
    props = compute_properties_image_set(cae_subjects, histEq=True)

    # train partition
    trainlen = (len(props) * 2) // 3
    trainpartition = props[0:trainlen]
    # augment train set
    # generate crops
    crops = []
    for imageset in trainpartition:
        for image in imageset:
            crops.extend(crop_image(image, wcrop=200))

    # flip crops
    crops = flip_image(crops)
    crops_roi = extract_roi_image_set(crops)
    make_lmdb_utfvp(crops_roi, dbpath + '/train', mode='train')

    # validation part
    valpartition = props[trainlen:]
    valcrops = []
    for imageset in valpartition:
        for image in imageset:
            valcrops.extend(crop_image(image, wcrop=200))

    # generate database
    patches = extract_roi_image_set(valcrops)
    make_lmdb_utfvp(patches, dbpath=dbpath + '/val', mode='train')

def generate_lmdb_utfvp_eval(datapath, dbpath, nSubject=40, nObjects=3):
    # read images
    evalsubs = read_images_UT(datapath, mode='eval', numSample=nSubject)
    evalsubs = compute_properties_image_set(evalsubs, histEq=True)

    # generate image pairs
    matedpairs, nonmatedpairs = generate_image_pairs(evalsubs, num_object=nObjects)

    # mated database
    db_path = dbpath + '/mated/'
    make_lmdb_utfvp(matedpairs, db_path, mode='eval')

    # non-mated database
    db_path = dbpath + '/nonmated/'
    make_lmdb_utfvp(nonmatedpairs, db_path, mode='eval')

def extract_roi_image_set(imageset):
    for i, img in enumerate(imageset):
        imageset[i] = cv2.resize(extract_roi(img), (256, 128), interpolation=cv2.INTER_AREA)
    return imageset

def generate_image_pairs(partition, num_object=3):
    # generate imposter and genuine pairs
    # num_object - number of imposter object images
    #              default - 3

    # genuine pairs
    genuinepairs = generate_genuine_pairs(partition)
    # imposter pairs
    imposterpairs = []
    # loop over all images
    for refId in range(len(partition)):
        set = partition[refId]
        for refimage in set:
            object_images = select_object_images(partition, refId, num_object)
            for objimage in object_images:
                if refimage.p == objimage.p and refimage.f == objimage.f:
                    print('Genuine in imposter set!!!!')
                objimage = register_images(refimage, objimage)
                # strip images from image set add label
                refImage = extract_roi(refimage.imageNormalised)
                objImage = extract_roi(objimage.imageRegistered)
                #if mode == 'eval':
                refImage = cv2.resize(refImage, (256, 128),
                              interpolation=cv2.INTER_CUBIC)
                objImage = cv2.resize(objImage, (256, 128),
                              interpolation=cv2.INTER_CUBIC)
                imposterpairs.append([refImage, objImage, 0])

    return genuinepairs, imposterpairs