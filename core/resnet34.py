import os
import glob
import re 
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn import preprocessing
import datetime
import time

import sys
sys.path.append("./dataset")  

from gaa import *
from single import *

class GAAResNet34():
    def __init__(self, output_classes=None, train_ratio=0.7, batch_size=32, epochs=5, verbose=True, pretrained_weight_file=None):
        self.model = resnet34(pretrained=True)
        #self.model = resnet34(pretrained=False)
        #self.model.fc = nn.Linear(512,35)
        self.model.fc = nn.Linear(512,output_classes)
        
        self.device = torch.device("cpu")
        self.model.cpu()
        self.verbose = verbose

        self.best_avg_loss = 100000000000000 #tekitou

        if pretrained_weight_file is not None:
            self.load(pretrained_weight_file)

    def train_aux(self,epoch):
        total_loss = 0
        total_size = 0
        self.model.train()
        report_percent = 10
        report = int((len(self.train_loader.dataset) / self.batch_size) * float(report_percent) / 100.0)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            s_t = time.time()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            self.optimizer.step()
            e_t = time.time()

            if self.verbose: 
                print("DEBUG: time=%d, batch_idx=%d, len(data)=%d, batch_idx * len(data)=%d" % (int(e_t-s_t),batch_idx, len(data), batch_idx*len(data)))
            if batch_idx % report == 0:
                now = datetime.datetime.now()
                avg_loss = total_loss / total_size
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                    now,
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx * len(data) / len(self.train_loader.dataset), avg_loss))

                if self.best_avg_loss > avg_loss:
                    print("BEST LOSS UPDATED!!!")
                    self.best_avg_loss = avg_loss
                    self.save("./weights/best_weight.pth")


            sys.stdout.flush()

    def setup_dataset_loader(self, dataset, batch_size, train_ratio):
        n_samples = len(dataset)
        train_size = int(len(dataset) * train_ratio)
        val_size = n_samples - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    def train(self, dataset, train_ratio=0.7, batch_size=32, epochs=5):
        print("INFO: train start. show model info")
        print(self.model)
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs

        self.setup_dataset_loader(dataset, self.batch_size, train_ratio)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(self.epochs):
            self.train_aux(epoch)

    def save(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()

    def test(self,dataset, batch_size=32, train_ratio=0.7):
        pred = []
        Y = []
        self.setup_dataset_loader(dataset, batch_size, train_ratio)
        for i, (data, target) in enumerate(self.val_loader):
            with torch.no_grad():
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
            pred += [int(l.argmax()) for l in output]
            Y += [int(l) for l in target]

        print(classification_report(Y, pred))

    def single_predict(self,file_name):
        im = SingleDataImageLoader()
        data = im.load(file_name)
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        max_idx = int(probabilities.argmax())
        score   = float(probabilities[max_idx])
        return max_idx, score




if __name__ == "__main__":

    class MyDataSet(Dataset):
        def __init__(self):
            l = glob.glob('../ed3c566ec089ca987ebc079b23937d9c/images/*.jpg')
            self.train_df = pd.DataFrame()
            self.images = []
            self.labels = []
            self.le = preprocessing.LabelEncoder()
    
            for path in l:
                self.images.append(path)
                self.labels.append(re.split('[/_.]', path)[5])
    
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


    import sys
    print("INFO main")
    #dataset = MyDataSet()
    dataset = GAADataSet()

    print("dataset size = %d" % (len(dataset)))
    print("dataset classses = %d" % (dataset.classes()))

    #gaa_resnet_34 = GAAResNet34(output_classes=dataset.classes(), verbose=False, pretrained_weight_file="./weights/resnet_only_try9.pth")
    gaa_resnet_34 = GAAResNet34(output_classes=dataset.classes(), verbose=False)
    if sys.argv[1] == "train":
        gaa_resnet_34.train(dataset,epochs=20)
        gaa_resnet_34.save("./weights/best_weight.pth")
    elif sys.argv[1] == "test":
        gaa_resnet_34.load("./weights/best_weight.pth")
        gaa_resnet_34.test(dataset)
    elif sys.argv[1] == "single":
        gaa_resnet_34.load("./weights/best_weight.pth")
        print(gaa_resnet_34.single_predict(sys.argv[2]))
    elif sys.argv[1] == "labels":
        dataset.print_labels()
    else:
        print("ERROR: invalid arg")
