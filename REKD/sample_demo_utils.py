
import numpy as np
import os, cv2, torch, config
import matplotlib.pyplot as plt
import glob

#from REKD.evaluation.extract_hpatches import MultiScaleFeatureExtractor
from training.model.REKD import REKD, count_model_parameters

def load_sample_model(args, model_path):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    args.load_dir =  model_path


    model = REKD(args, device)
    model.load_state_dict(torch.load(args.load_dir))  ## Load the PyTorch learnable model parameters.

    model.export()
    model.eval()
    model = model.to(device) ## use GPU


    print("Model paramter : {} is loaded.".format( args.load_dir ))
    count_model_parameters(model)
    return model

def load_images():



def model_output():

def apply_NMF():

def finalfeatures_REKD(model):
    #do the final layer features semanically correspond to a vector field?? --> can they point to the center??
    #KL divergence or L2 norm?? --> we want similarity
    return 