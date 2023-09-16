import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def evaluate(net: torch.nn, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    'evaluates model accuracy on loader data'
    # Evaluate
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total



def train_model(net, tr_loader, te_loader, device, criterion, optimizer, num_epochs: int, frequency: int = 2000) -> tuple[list, list, list]:
    'trains model with parameters and returns loss, tr_accuracy, te_accuracy'

    print('starting training')

    loss_vals, tr_acc, te_acc = [], [], []

    running_loss = 0.0

    # train the model 
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tr_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # backward + optimze
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % frequency == frequency-1:    # print every 2000 mini-batches
                tr_a = evaluate(net, tr_loader, device)
                te_a = evaluate(net, te_loader, device)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / frequency:.3f} tr acc: {tr_a:.3f} te acc: {te_a:.3f}')

                # add vals to lists
                loss_vals.append(running_loss / frequency)
                tr_acc.append(tr_a)
                te_acc.append(te_a)

                running_loss = 0.0
    
    tr_a = evaluate(net, tr_loader, device)
    te_a = evaluate(net, te_loader, device)
    loss_vals.append(running_loss)
    tr_acc.append(tr_a)
    te_acc.append(te_a)

    print('Finished Training')
    return loss_vals, tr_acc, te_acc


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_errors(net, loader, classes, num_row_out):
    cur = 0

    for images, labels in loader:
        # print images
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        if (predicted == labels).sum().item() == 0:
            imshow(torchvision.utils.make_grid(images))
            print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

            

            print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                        for j in range(4)))
            cur+=1

            if cur > num_row_out:
                break