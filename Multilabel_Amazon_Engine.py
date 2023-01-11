from tqdm import tqdm
import torch.nn as nn
from torch import no_grad, save
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import os
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, f1_score, classification_report
import seaborn as sns
from datetime import date

tags = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
        'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming',
        'selective_logging', 'blow_down']


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


def calculate_metrics(pred, target, batch_loss=None):
    """
    Calculates the following metrics: (precision, recall, f1) with micro, macro and samples averaging.
    Additionally, hamming loss is also calculated.
    For some later use, batch loss, as calculated per chosen criterion, is also added
    :param pred: prediction array
    :param target: targeted value (ground truth)
    :param batch_loss: batch_loss
    :return: dictionary with keys: "X/precision", "X/recall", "X/f1" with X='micro', 'macro' or 'samples',
    and also 'hamming_loss'
    """
    results = {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
               'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
               'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
               'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
               'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
               'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
               'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples', zero_division=0),
               'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples', zero_division=0),
               'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples', zero_division=0),
               'hamming_loss': hamming_loss(y_true=target, y_pred=pred),
               'total_loss': batch_loss
               }
    return results


def append_metrics(all_metrics_dict, batch_metrics):
    """
    Appends the batch measured metrics to whole batch metrics
    :param all_metrics_dict: dictionary with all the metrics saved
    :param batch_metrics: dictionary with same metrics but only for the batch
    :return: the updated dictionary with all lists of metrics
    """
    for key in batch_metrics.keys():
        metric_list = all_metrics_dict[key]
        metric_list.append(batch_metrics[key])
        all_metrics_dict[key] = metric_list
    return all_metrics_dict


def append_mean_metrics(all_metrics_dict, batch_metrics):
    """
    Appends the mean value for each metric
    :param all_metrics_dict:
    :param batch_metrics:
    :return:
    """
    for key in batch_metrics.keys():
        overall_metric_list = all_metrics_dict[key]
        overall_metric_list.append(np.mean(batch_metrics[key]))
        all_metrics_dict[key] = overall_metric_list
    return all_metrics_dict


def validate(model, dataloader, device, loss_fn=nn.BCEWithLogitsLoss()):
    """
    Function for loop over validation dataloader and measuring the metrics
    :param model: model to validate
    :param dataloader: validation dataloader
    :param device: device : "cuda" or "cpu"
    :param loss_fn: default = BCEWithLogitsLoss()
    :return: 5 numpy arrays for : total loss array, all accuracy array, precision score array,recall score array
    """
    sig = nn.Sigmoid()

    model.eval()

    # accs, acc_scores, prec_scores, rec_scores, tot_loss, ham_loss = [], [], [], [], [], []
    overall_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                       'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                       'samples/f1': [], 'hamming_loss': [], 'total_loss': []}

    print('Validating')
    with no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader)):
            image_batch = sample_batch['image'].to(device)
            targets = sample_batch['labels'].to(device)

            # Model Predictions
            out = model(image_batch)

            # Get loss (Sigmoid + Cross Entropy function)
            loss = loss_fn(out, targets)
            # Appending it to all the losses
            # tot_loss.append(loss.cpu().detach().item())

            # apply sigmoid activation to get all the outputs between 0 and 1
            predicted = (sig(out) > 0.5).float().cpu().detach().numpy()
            ground_truth = targets.cpu().detach().numpy()

            # save metrics
            batch_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().item())

            # Append metrics to the overall epoch metrics measures
            append_metrics(overall_metrics, batch_metrics)

    # return tot_loss, accs, acc_scores, prec_scores, rec_scores, ham_loss
    return overall_metrics


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
    optimizer = optimizer or Adam(model.parameters(), lr=lr)
    model.train()

    epoch_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                     'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                     'samples/f1': [], 'hamming_loss': [], 'total_loss': []}

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
        # Prediction of this batch and appending to all accuarcies of this epoch
        predicted = (sig(out) > 0.5).float().cpu().detach().numpy()
        ground_truth = targets.cpu().detach().numpy()

        # save all the metrics
        batch_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().item())

        # Append metrics to the overall epoch metrics measures
        append_metrics(epoch_metrics, batch_metrics)

        if i_batch == 0:
            print(f'image batch size: {image_batch.size()}')
            print(f' Predicted shape: {np.shape(predicted)} and ground truth shape {np.shape(ground_truth)}')
            print(batch_metrics)
            print("iter:{:3d} training:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}"
                  "loss: {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                        batch_metrics['samples/f1'], batch_metrics['total_loss']))
            print("Predicted:")
            print(predicted[range(4), :])
            print("Ground-truth")
            print(ground_truth[range(4), :])
            show_4_image_in_batch(image_batch, predicted_labels=predicted, ground_truth=ground_truth)
            continue

        if i_batch % 100 == 0:  # print every ... mini-batches the mean loss up to now
            print("iter:{:3d} training:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}"
                  "loss: {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                        batch_metrics['samples/f1'], batch_metrics['total_loss']))

        if i_batch % 500 == 0:
            show_4_image_in_batch(image_batch, predicted_labels=predicted, ground_truth=ground_truth)

    return epoch_metrics


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
    :return: results as a dictionary with 'train_loss', 'train_acc', 'train_acc_scores', 'train_prec_scores',
     'train_rec_scores', 'train_ham_loss', 'val_loss', 'val_acc', 'val_acc_scores', 'val_prec_scores',
     'val_rec_scores', 'val_ham_loss'
    """
    optimizer = optimizer or Adam(model.parameters(), lr=lr)
    # res = {'train_loss': [], 'train_acc': [], 'train_acc_scores': [], 'train_prec_scores': [], 'train_rec_scores': [],
    #       'train_ham_loss': [],
    #       'val_loss': [], 'val_acc': [], 'val_acc_scores': [], 'val_prec_scores': [], 'val_rec_scores': [],
    #       'val_ham_loss': []}

    overall_metrics = {'training': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                                    'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                                    'samples/f1': [], 'hamming_loss': [], 'total_loss': [], 'report': []},
                       'validating': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                                      'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                                      'samples/f1': [], 'hamming_loss': [], 'total_loss': []}
                       }

    min_loss = 1000

    for ep in range(epochs):
        print(f'Training epoch {ep} ..... ')

        epoch_metrics = train_epoch(model, train_loader, optimizer=optimizer, lr=lr, loss_fn=loss_fn,
                                    device=device)
        val_metrics = validate(model, validation_dataloader, loss_fn=loss_fn, device=device)

        # check if best performance, save if yes
        if np.mean(val_metrics['total_loss']) < min_loss:
            min_loss = np.mean(val_metrics['total_loss'])
            model_save_name = f"model_multilabel_{epochs}epochs_{str(lr).replace('.','_')}lr_{date.today()}.pth"
            save(model.state_dict(), model_save_name)
            print(f"Saved PyTorch Model State to {model_save_name}")

        # Append all the metrics
        append_mean_metrics(overall_metrics['training'], epoch_metrics)
        append_mean_metrics(overall_metrics['validating'], val_metrics)

    print(".... ENDED TRAINING THE MODEL !")
    return overall_metrics


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
    predicted = (sig(y_hat) > 0.5).float().cpu().detach().numpy()
    ground_truth = y.cpu().detach().numpy()

    predictions = np.array((predicted == ground_truth), dtype=np.float64).mean(axis=0)
    accuracy = (np.array((predicted == ground_truth)).astype(np.float64).mean())

    return loss.cpu().detach().numpy(), accuracy, predictions


def show_4_image_in_batch(images_batch, predicted_labels, ground_truth):
    """
    Shows 4 first images from the batch of the Amazon Dataset
    :param sample_batched: mini-batch of dataloader of Amazon Dataset. Dictionary with 'image', 'labels'
    :param tags: All the unique labels
    :return:
    """
    tags = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
            'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming',
            'selective_logging', 'blow_down']
    num_tags = np.arange(start=0, stop=len(tags))

    # images_batch, labels = sample_batched['image'], sample_batched['labels']

    fig, axs = plt.subplots(1, 4, sharey=True)
    for i in range(4):
        img = F.to_pil_image(images_batch[i])
        axs[i].imshow(img)
        axs[i].grid(False)
        axs[i].set_axis_off()
        ids = num_tags[predicted_labels[i, :] == 1.0]
        ids_truth = num_tags[ground_truth[i, :] == 1]
        names = [tags[ix] for ix in ids]
        names_truth = [tags[i] for i in ids_truth]
        axs[i].set_title(f'#{i}:\n pred: {names} \n truth: {names_truth}', {'fontsize': 12})
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.tight_layout()
    plt.show()


def batch_prediction_s(batch, model, device="cuda", criterion=nn.BCEWithLogitsLoss()):
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
    predicted = (sig(y_hat) > 0.5).float().cpu().detach().numpy()
    ground_truth = y.cpu().detach().numpy()

    predictions = np.array((predicted == ground_truth), dtype=np.float64).mean(axis=0)

    return loss.cpu().detach().numpy(), predicted, ground_truth, predictions


def compute_metrics(test_dataloader, model, device, tags):
    """
    Predict the values from model for test_dataloader given. Function for the testing of the model.
    :param test_dataloader:
    :param model: model of interest
    :param device: on which device run the thing
    :param tags: name of the classes
    :return: report, losses
    """

    # store stats
    losses = []
    count = 0

    for batch in tqdm(test_dataloader):
        # TODO run prediction_step
        loss, predicted, ground_truth, predictions = batch_prediction_s(batch, model, device=device)

        # append to stats
        losses.append(loss)

        # accuracies.append(accuracy)
        if count == 0:
            all_predicted = predicted
            all_truth = ground_truth
            all_prediction = predictions
            count = 1
        else:
            all_predicted = np.vstack((all_predicted, predicted))
            all_truth = np.vstack((all_truth, ground_truth))
            all_prediction = np.vstack((all_prediction, predictions))

    report = classification_report(y_true=all_truth, y_pred=all_predicted, output_dict=True, target_names=tags,
                                   zero_division=0)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="mako")
    return report, losses, all_prediction
