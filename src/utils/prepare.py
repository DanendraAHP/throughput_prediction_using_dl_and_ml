import os
from src.common.constant import PATH

def prepare_folder():
    #check csv folder
    if not os.path.isdir(PATH.csv) :
        os.mkdir(PATH.csv)

    #check model folder
    if not os.path.isdir(PATH.csv) :
        os.mkdir(PATH.model)