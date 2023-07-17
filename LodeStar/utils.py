import numpy as np
import scipy

def rototranslate_input(translate, rotate, input):
        # if isinstance(image, np.array):
        #     input = image.numpy()

        cr = np.cos(rotate)
        sr = np.sin(rotate)

        mapping = np.array([[cr, sr], [-sr, cr]])

        dx, dy = translate
        
        shape = input.shape
        center = np.array(shape[:2]) / 2

        d = center - np.dot(mapping, center) - np.array([dy, dx])
        # print(mapping)

        new_image = scipy.ndimage.affine_transform(
                input=input,
                matrix=mapping,
                offset=d
            )
        return new_image

def inverse_point(translate, rotate, input):
        