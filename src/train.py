import PIL
import os
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import scipy


class ArtDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform=None):
        self.df = pd.read_csv(csv_path, sep=",", header=None, names=["filename", "label"])
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df['filename'][index]
        label = self.df['label'][index]
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, trainloader, optimizer, criterion, PATH, epochs=20):
    print(len(trainloader))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(epochs):  # loop over the dataset multiple times
        # running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            print(f'[{epoch + 1}, {(i + 1) / len(trainloader) * 100:.2f}%] loss: {loss.item():.3f}')
            # if i % 50 == 49:    # print every 10 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            #     running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), PATH)


def test(model, classes, testloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    i = 0

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label.item()]] += 1
                total_pred[classes[label.item()]] += 1
            print(f'{(i + 1) / len(testloader) * 100:.1f}% done testing')
            i += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def class_to_label(label_class, labels) -> torch.Tensor:
    df = pd.read_csv(label_class, sep=" ", header=None, names=["label", "class"])
    res = []
    for label in labels:
        res.append(df['class'][label.item()])
    return res


def main():
    transform = torchvision.transforms.Compose([
        # you can add other transformations in this list
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    goal = "style"
    batch_size = 32
    num_workers = 6
    lr = 0.001
    epochs = 6

    train_set = ArtDataset('../data/wikiart_csv/{goal}_train.csv'.format(goal=goal), '../data/wikiart', transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = ArtDataset('../data/wikiart_csv/{goal}_val.csv'.format(goal=goal), '../data/wikiart', transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    classes = pd.read_csv("../data/wikiart_csv/{goal}_class.txt".format(goal=goal), sep=" ", header=None)[1]

    model = NeuralNetwork(len(classes))

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    input = './dream_net_6ep_001lr.pth'
    output = './dream_net_test.pth'

    model.load_state_dict(torch.load(input))

    train(model, trainloader, optimizer, criterion, output, epochs=epochs)

    model.load_state_dict(torch.load(output))
    model.eval()

    test(model, classes, testloader)


if __name__ == '__main__':
    main()
