import argparse, os
from generate_datasets import generate_lmdb_utfvp_eval

# # parse command line arguments
# parser = argparse.ArgumentParser(description='Finger vein train set generation')
# parser.add_argument('datapath', help='Folder for raw images', nargs='+', type=str)
# parser.add_argument('lmdbpath', help='Folder for lmdb database', nargs='+', type=str)
# parser.add_argument('nSubjects', help='Number of training subjects(Def. 40)', default=40, type=int)
# parser.add_argument('nObjects', help='Number of nonmated pairs per image (Def. 3)', default=3, type=int)
# args = parser.parse_args()
# datapath = args.datapath[0]
# lmdbpath = args.lmdbpath[0]
# nSubjects = args.nSubjects


datapath = '../CAE/dataset/Twente/dataset/data/'
lmdbpath = './database/UTFVP_roi_eval/'
nObjects = 3
nSubjects = 1

if not os.path.exists(lmdbpath):
    os.mkdir(lmdbpath)

def main():
    generate_lmdb_utfvp_eval(datapath, lmdbpath, nSubject=nSubjects, nObjects=nObjects)

if __name__ == '__main__':
    main()