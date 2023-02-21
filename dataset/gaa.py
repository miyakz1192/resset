# Game Ad Automation data loader

import re
import os
import pickle
import glob
from PIL import Image
import torchvision.transforms as transforms
from sklearn import preprocessing

class GAADataSet:
    IMAGE_PATH     = "./dataset/GAA_DATA/data_set/JPEGImages/"
    SAVE_PATH      = "./dataset/GAA_DATA/gaa_data.pkl"
    CLOSE_LABEL = 1000

    def __init__(self):
        l = glob.glob(self.IMAGE_PATH+"*.jpg")
        self.images = []
        self.labels = []

#        l = self.cut_data(l)

        self.le = preprocessing.LabelEncoder()
        
        for path in l:
            self.images.append(path)
            self.labels.append(self.process_ja_char(path) + self.process_close(path))
        
        self.le.fit(self.labels)
        self.labels_id = self.le.transform(self.labels)
		#for train
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        #old
        #self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def cut_data(self,l):
        ja_char = [s for s in l if re.match('.*ja_char.*\.jpg', s)]
        closew  = [s for s in l if re.match('.*closew.*\.jpg' , s)]

        return closew + ja_char

    def process_ja_char(self, item):
        res = re.match(self.IMAGE_PATH+"ja_char_(?P<label>\d+)_(?P<number>\d+)", item)
        if res is None:
            return ""
    
        label  = int(res.group("label"))

        return str(label)

    def process_close(self, item):
        res = re.match(self.IMAGE_PATH+"ja_char_(?P<label>\d+)_(?P<number>\d+)", item)
        if res is not None:
            return ""

        s = item.split("/") #split with directory char "/"(ex:./data_set/JPEGImages/closegb_201.jpg)
        f = s[len(s)-1]     #get file name (ex: closegb_201.jpg)
        return "".join(reversed("_".join("".join(reversed(f)).split("_")[1:]))) #return label (ex: closegb)

    def print_labels(self):
        for i in range(len(self.images)):
            print(self.labels[i], self.labels_id[i])

    def classes(self):
        return len(list(set(self.labels_id)))

    def label_ids(self):
        return dict(zip(self.labels, self.labels_id))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.convert('RGB')
        label = self.labels_id[idx]
        return self.transform(image), int(label)
