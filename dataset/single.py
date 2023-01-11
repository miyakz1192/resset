#single image data loader

from PIL import Image
import torch
import torchvision.transforms as transforms

class SingleDataImageLoader:

    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def load(self, file_name):
        image = Image.open(file_name)
        image = image.convert('RGB')
        image = self.transform(image)
        image = torch.unsqueeze(image, axis=0)
        return image
