import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

## input: a single image
## output : rotation angles and rotates images

#Can't use dataloader --> you need to know the rotations and you need to reverse them



#should return an output image
# **kwargs??


    
#numpy array 

# return a batch??
# return a single translated image??
# store rototranslation batches
# dataloader --> creates rototranslations, applies them on the image, and returns a batch on the 


#include static method of rotating an image
class RotoTranslation:
    def __init__(self, translate, rotate):
        self.rotate = rotate
        self.translate = translate
        
        cr = np.cos(self.rotate)
        sr = np.sin(self.rotate)

        self.mapping = np.array([[cr, sr], [-sr, cr]])


    def rototranslate(self, image):
        # if isinstance(image, np.array):
        #     input = image.numpy()

        dx, dy = self.translate
        
        shape = input.shape
        center = np.array(shape[:2]) / 2

        d = center - np.dot(self.mapping, center) - np.array([dy, dx])

        new_image = scipy.ndimage.affine_transform(
                input=input,
                matrix=self.mapping,
                offset=d
            )
        return new_image
    
    def inverse(self, image, translate):

        if isinstance(image) == torch.tensor:
            input = image.numpy()

        dx, dy = self.translate
        
        shape = image.shape
        center = np.array(shape[:2]) / 2

        d = center - np.dot(self.mapping, center) - np.array([dy, dx])
        inverse_mapping = np.linalg.inv(self.mapping)

        new_image = scipy.ndimage.affine_transform(
                input=image,
                matrix=self.mapping,
            )
        return new_image



# #kth batch should have a kth rototranslate --> this should lead to exact reversal, mean, and then a loss calculation
# class DataConstructor:
#     def __init__(self, ):

#     def __getitem__(self, ):
        

# def loss(batch, rototranslations):
#     output = model(batch)
#     for k in output.shape[0]:
#         output[k,:]
#         original = rototranslations[k].inverse()
#         # l1 dist(output, original)


class Lodestar:
    def __init__(self, N, M): #parameters: input tensor shape, output shape, make a 
        self.output = OutputHead #this output head can be of any type --> provides an interface

#abstract class
class OutputHead:
    def __init__(self):
        
    def output_tensor(self, ):


class 