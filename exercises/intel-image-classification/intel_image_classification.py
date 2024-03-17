import sys

import torch
import torch.nn.functional as F 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import models
from torchvision.datasets import ImageFolder

from PIL import Image

import numpy as np

from IPython import embed


def evaluate(model, test_loader):    
    model.eval()
    accuracy = 0
    with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
        for X, y in test_loader:
            y_hat = model(X)
            y_hat = F.softmax(y_hat, dim=1).detach().numpy()
            y_hat = np.argmax(y_hat, axis=1)
            accuracy += (y_hat == y.detach().numpy()).mean()
    accuracy /= len(test_loader)  # avg accuracy

    return accuracy


def fit(epochs, model, optimizer, train_dl):
    loss_func = nn.CrossEntropyLoss()

    # loop over epochs
    for epoch in range(epochs):
        model.train()

        # loop over mini-batches
        for X_mb, y_mb in train_dl:
            y_hat = model(X_mb)

            loss = loss_func(y_hat, y_mb)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

        model.eval()

    print('Finished training')

    return model


def finetune(model, train_dl):
    for param in model.parameters():  # Freeze parameters so we don't update them
        param.requires_grad = False

    new_layers = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=9216, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=1000, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=1000, out_features=6, bias=True),
        # nn.Softmax(dim=1)
    )
    model.classifier = new_layers

    optimizer = optim.Adam(model.parameters())

    epochs = 10
    trained_model = fit(epochs, model, optimizer, train_dl)

    return trained_model


def main(args):
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),   # resize the input to 224x224
        transforms.ToTensor(),          # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input based on images from ImageNet
    ])

    alexnet = models.alexnet(pretrained=True)

    # finetune
    train_ds = ImageFolder(root="seg_train", transform=data_transforms)
    mini_batch_size = 512
    train_dl = DataLoader(train_ds, batch_size=mini_batch_size, shuffle=True, drop_last=False)
    trained_model = finetune(alexnet, train_dl)

    # test image
    # put the model to eval mode for testing
    # alexnet.eval()
    # output = alexnet(batch_img)
    trained_model.eval()

    test_image = Image.open("seg_test/sea/21191.jpg")
    print("original image's shape: " + str(test_image.size))
    transformed_img = data_transforms(test_image)
    print("transformed image's shape: " + str(transformed_img.shape))
    # form a batch with only one image
    batch_img = torch.unsqueeze(transformed_img, 0)
    print("image batch's shape: " + str(batch_img.shape))

    output = trained_model(batch_img)

    print("output vector's shape: " + str(output.shape))
    percentage = F.softmax(output, dim=1)[0] * 100.0
    _, indices = torch.sort(output, descending=True)

    # map the class no. to the corresponding label
    with open('class_names_Intel.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]
    results = [(classes[i], percentage[i].item()) for i in indices[0][:5]]

    for i in range(3):
        print('{}: {:.4f}%'.format(results[i][0], results[i][1]))

    # test run
    test_ds = ImageFolder(root="seg_test", transform=data_transforms)
    mini_batch_size = 512 * 2
    test_dl = DataLoader(test_ds, batch_size=mini_batch_size)
    accuracy = evaluate(trained_model, test_dl)
    print("accuracy: " + str(accuracy))

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
