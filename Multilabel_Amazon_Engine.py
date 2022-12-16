from tqdm import tqdm
import torch.nn as nn
import torch.no_grad
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss


def checking_folder(data_folder='../IPEO_Planet_project'):
    labels_dt = pd.read_csv(f'{data_folder}/train_labels.csv', dtype=str)
    corrupted_files = []
    i = 0
    for img_id in tqdm(labels_dt['image_name']):
        # if i % 1000 == 0:
        #    print(f"image {i}")
        img_name = os.path.join(f'{data_folder}/train-jpg', f'{img_id}.jpg')
        img = imread(img_name)
        if img is None:
            print(f'{img_id} is corrupt...')
            corrupted_files.append(img_id)
        img = None
        i = i + 1
    return corrupted_files


def validate(model, dataloader, device, loss_fn=nn.BCEWithLogitsLoss()):
    '''

    :param model: model to validate
    :param dataloader: validation dataloader
    :param device: device : "cuda" or "cpu"
    :param loss_fn: default = BCEWithLogitsLoss()
    :return: 5 numpy arrays for : total loss array, all accuracy array, precision score array,recall score array
    '''
    sig = nn.Sigmoid()
    model.eval()
    accs, acc_scores, prec_scores, rec_scores, tot_loss, ham_loss = [], [], [], [], [], []
    print('Validating')
    with torch.no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader)):
            image_batch = sample_batch['image'].to(device)
            targets = sample_batch['labels'].to(device)

            # Model Predictions
            out = model(image_batch)

            # Get loss (Sigmoid + Cross Entropy function)
            loss = loss_fn(out, targets)
            # Appending it to all the losses
            tot_loss.append(loss.cpu().detach().numpy())

            # apply sigmoid activation to get all the outputs between 0 and 1
            predicted = (sig(out) > 0.5).float().cpu().detach().numpy()
            ground_truth = targets.cpu().detach().numpy()

            # save metrics
            accs.append(np.mean(np.array(predicted == ground_truth),axis=0))
            acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
            prec_scores.append(precision_score(ground_truth.flatten(),predicted.flatten()))
            rec_scores.append(recall_score(ground_truth.flatten(),predicted.flatten()))
            ham_loss.append(hamming_loss(ground_truth.flatten(),predicted.flatten()))

    return tot_loss, accs, acc_scores, prec_scores, rec_scores, ham_loss


def train_epoch(model, dataloader, device, lr=0.01, optimizer=None, loss_fn=nn.BCEWithLogitsLoss()):
    """
    :param model: model used for prediction
    :param dataloader: dataloader over the amazon space dataset
    :param device: device on which train, 'cuda' or 'cpu'
    :param lr: learning rate, here default = 0.01
    :param optimizer: Here if None given then using Adam
    :param loss_fn: Loss function default is BCEWithLogitsLoss (sigmoid + cross-entropy)
    :return:
    """
    sig = nn.Sigmoid()
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    model.train()
    accs, acc_scores, prec_scores, rec_scores, tot_loss, ham_loss = [], [], [], [], [], []
    print('Training')
    for i_batch, sample_batch in tqdm(enumerate(dataloader)):
        # get the inputs.
        image_batch = sample_batch['image'].to(device)
        targets = sample_batch['labels'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Prediction from model
        out = model(image_batch)

        # Loss function -> sigmoid included in BCEWithLogistLoss
        loss = loss_fn(out, targets)

        # backpropagation
        loss.backward()

        # update optimizer parameters
        optimizer.step()

        # Metrics
        # All the losses for this epoch
        tot_loss.append(loss.cpu().detach().numpy())
        # Prediction of this batch and appending to all accuarcies of this epoch
        predicted = (sig(out) > 0.6).float().cpu().detach().numpy()
        ground_truth = targets.cpu().detach().numpy()

        #save all the metrics
        accs.append(np.mean(np.array(predicted == ground_truth),axis=0))
        acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
        prec_scores.append(precision_score(ground_truth.flatten(),predicted.flatten()))
        rec_scores.append(recall_score(ground_truth.flatten(),predicted.flatten()))
        ham_loss.append(hamming_loss(ground_truth.flatten(),predicted.flatten()))

        if i_batch == 0:
            print(image_batch.size())
            print(np.shape(predicted), np.shape(ground_truth))
            print(f"Predicted : {predicted}, calculated accuracy score: {np.mean(acc_scores)}, prediction score : {np.mean(prec_scores)}, recall score: {np.mean(rec_scores)}")

        if i_batch % 20 == 0:  # print every ... mini-batches the mean loss up to now
            print(f"Loss : {np.mean(tot_loss)}, calculated accuracy score: {np.mean(acc_scores)}, prediction score : {np.mean(prec_scores)}, recall score: {np.mean(rec_scores)}")

    return tot_loss, accs, acc_scores, prec_scores, rec_scores, ham_loss


def train(model, train_loader, validation_dataloader, device, optimizer=None, lr=0.01, epochs=2,
          loss_fn=nn.BCEWithLogitsLoss()):
    """
    :param model: model to train
    :param train_loader: train dataloader
    :param test_loader: testing dataloader
    :param optimizer: optimizer per default is Adam
    :param lr: default 0.01
    :param epochs: default 2
    :param loss_fn: default BCEWithLogitsLoss (sigmoid + Cross Entropy)
    :return: results as a dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    res = {'train_loss': [], 'train_acc': [], 'train_acc_scores':[], 'train_pred_scores':[], 'train_rec_scores':[],
           'train_ham_loss': [],
           'val_loss': [], 'val_acc': [],'val_acc_scores': [], 'val_pred_scores': [], 'val_rec_scores': [], 'val_ham_loss':[]}
    for ep in range(epochs):
        tl, ta, a_s, p_s, r_s, h_l = train_epoch(model, train_loader, optimizer=optimizer, lr=lr, loss_fn=loss_fn,
                                                 device=device)
        vl, va, va_s, vp_s, vr_s, vh_l = validate(model, validation_dataloader, loss_fn=loss_fn, device=device)

        print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['train_acc_scores'].append(a_s)
        res['train_pred_scores'].append(p_s)
        res['train_rec_scores'].append(r_s)
        res['train_ham_loss'].append(h_l)

        res['val_loss'].append(vl)
        res['val_acc'].append(va)
        res['val_acc_scores'].append(va_s)
        res['val_pred_scores'].append(vp_s)
        res['val_rec_scores'].append(vp_s)
        res['val_ham_loss'].append(vh_l)
    return res


def show_4_image_in_batch(sample_batched, tags):
    """
    Shows 4 first images from the batch of the Amazon Dataset
    :param sample_batched: mini-batch of dataloader of Amazon Dataset. Dictionary with 'image', 'labels'
    :param tags: All the unique labels
    :return:
    """
    images_batch, labels = sample_batched['image'], sample_batched['labels']

    fig, axs = plt.subplots(1, 4)
    for i in range(4):
        img = F.to_pil_image(images_batch[i])
        axs[i].imshow(img)
        ids = labels[i, :].numpy()
        axs[i].set_title(f'#{i}:\n {tags[ids == 1]}')
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.tight_layout()
    plt.show()


def batch_prediction(batch, model, device="cuda", criterion=nn.BCEWithLogitsLoss()):
    """
    Predict the values from model for batch given. Function for the testing of the model.
    :param criterion:
    :param batch: batch composed of 'image' and 'labels'
    :param model: model of interest
    :param device: on which device run the thing
    :return: loss,accuracy
    """
    model.eval()
    sig = nn.Sigmoid()

    # Retrieve image and label from the batch
    x = batch['image']
    y = batch['labels']

    # move model and code to GPU
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    # Forward pass
    y_hat = model(x)

    # Loss calculation (only for statistics)
    loss = criterion(y_hat, y)

    # Calculate accuracy for statistics
    predicted = (sig(y_hat) > 0.8).float().cpu().detach().numpy()
    ground_truth = y.cpu().detach().numpy()
    predictions = np.array((predicted == ground_truth), dtype=np.float64).mean(axis=0)
    accuracy = (np.array((predicted == ground_truth)).astype(np.float64).mean())

    return loss.cpu().detach().numpy(), accuracy, predictions

#%%
