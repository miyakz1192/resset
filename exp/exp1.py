import os
import glob
import re 
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn import preprocessing
import datetime


class MyDataSet(Dataset):
    def __init__(self):
        
        l = glob.glob('images/*.jpg')
        self.train_df = pd.DataFrame()
        self.images = []
        self.labels = []
        self.le = preprocessing.LabelEncoder()

        for path in l:
            self.images.append(path)
            self.labels.append(re.split('[/_.]', path)[1])

        self.le.fit(self.labels)
        self.labels_id = self.le.transform(self.labels)
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.convert('RGB')
        label = self.labels_id[idx]
        return self.transform(image), int(label)


dataset = MyDataSet()
n_samples = len(dataset)
train_size = int(len(dataset) * 0.7)
val_size = n_samples - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)


model = resnet34(pretrained=True)
model.fc = nn.Linear(512,37)

device = torch.device("cpu")
model.cpu()

model.load_state_dict(torch.load('../../ed3c566ec089ca987ebc079b23937d9c/sample_model_weights.pth'))
model.eval()


pred = []
Y = []
#braek one loop because of inspection
for i, (data, target) in enumerate(val_loader):
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        import pdb
        pdb.set_trace()
        output = model(data)
    pred += [int(l.argmax()) for l in output]
    Y += [int(l) for l in target]
    break

print(classification_report(Y, pred))
