
#include static method of rotating an image
class RotoTranslation:
    def __init__(self, translate, rotate):
        self.rotate = rotate
        self.translate = translate
        
        cr = np.cos(self.rotate)
        sr = np.sin(self.rotate)

        self.mapping = np.array([[cr, sr], [-sr, cr]])


    def rototranslate(self, input):
        # if isinstance(image, np.array):
        #     input = image.numpy()

        dx, dy = self.translate
        
        shape = input.shape
        center = np.array(shape[:2]) / 2

        d = center - np.dot(self.mapping, center) - np.array([dy, dx])
        print(self.mapping)

        new_image = scipy.ndimage.affine_transform(
                input=input,
                matrix=self.mapping,
                offset=d
            )
        return new_image
    
    def inverse(self, image):

        # if isinstance(image) == torch.tensor:
        #     input = image.numpy()

        dx, dy = self.translate
        
        shape = image.shape
        center = np.array(shape[:2]) / 2

        inverse_mapping = np.linalg.inv(self.mapping)
        print(inverse_mapping)

        d = center + np.dot(inverse_mapping, center) + np.array([dy, dx])


        new_image = scipy.ndimage.affine_transform(
                input=image,
                matrix=inverse_mapping,
                offset=d
            )
        return new_image
