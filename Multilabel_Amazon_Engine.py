from tqdm import tqdm
import torch.nn as nn
from torch import no_grad
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, f1_score


# from sklearn import metrics


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


def calculate_metrics(pred, target, batch_loss):
    """
    Calculates the following metrics: (precision, recall, f1) with micro, macro and samples averaging.
    Additionally, hamming loss is also calculated.
    For some later use, batch loss, as calculated per chosen criterion, is also added
    :param pred: prediction array
    :param target: targeted value (ground truth)
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

            # accs.append(np.mean(np.array(predicted == ground_truth), axis=0).tolist())
            # acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
            # prec_scores.append(precision_score(ground_truth.flatten(), predicted.flatten()))
            # rec_scores.append(recall_score(ground_truth.flatten(), predicted.flatten()))
            # ham_loss.append(hamming_loss(ground_truth.flatten(), predicted.flatten()))

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
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    # accs, acc_scores, prec_scores, rec_scores, tot_loss, ham_loss = [], [], [], [], [], []
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
        # All the losses for this epoch
        # tot_loss.append(loss.cpu().detach().item())
        # Prediction of this batch and appending to all accuarcies of this epoch
        predicted = (sig(out) > 0.5).float().cpu().detach().numpy()
        ground_truth = targets.cpu().detach().numpy()

        # save all the metrics
        batch_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().item())

        # accs.append(np.mean(np.array(predicted == ground_truth), axis=0).tolist())
        # acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
        # prec_scores.append(precision_score(ground_truth.flatten(), predicted.flatten()))
        # rec_scores.append(recall_score(ground_truth.flatten(), predicted.flatten()))
        # ham_loss.append(hamming_loss(ground_truth.flatten(), predicted.flatten()))

        # Append metrics to the overall epoch metrics measures
        append_metrics(epoch_metrics, batch_metrics)

        # if i_batch == 0:
        #     print(image_batch.size())
        #     print(np.shape(predicted), np.shape(ground_truth))
        #     print(
        #         f"Predicted : {predicted}, calculated accuracy score: {np.mean(acc_scores)}, prediction score : {np.mean(prec_scores)}, recall score: {np.mean(rec_scores)}")
        #
        # if i_batch % 20 == 0:  # print every ... mini-batches the mean loss up to now
        #     print(
        #         f"Loss : {np.mean(tot_loss)}, calculated accuracy score: {np.mean(acc_scores)}, prediction score : {np.mean(prec_scores)}, recall score: {np.mean(rec_scores)}")
        # return tot_loss, accs, acc_scores, prec_scores, rec_scores, ham_loss

        if i_batch == 0:
            print(f'image batch size: {image_batch.size()}')
            print(f' Predicted shape: {np.shape(predicted)} and ground truth shape {np.shape(ground_truth)}')
            print(batch_metrics)
            print("iter:{:3d} training:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                              batch_metrics['samples/f1']))
            print("Predicted:")
            print(predicted[[1, 15, 20, 36, 40], :])
            print("Ground-truth")
            print(ground_truth[[[1, 15, 20, 36, 40]], :])
            show_4_image_in_batch(image_batch, predicted_labels=predicted)
            continue

        if i_batch % 20 == 0:  # print every ... mini-batches the mean loss up to now
            print("iter:{:3d} training:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                              batch_metrics['samples/f1']))

        if i_batch % 60 == 0:
            show_4_image_in_batch(image_batch, predicted_labels=predicted)

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
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    # res = {'train_loss': [], 'train_acc': [], 'train_acc_scores': [], 'train_prec_scores': [], 'train_rec_scores': [],
    #       'train_ham_loss': [],
    #       'val_loss': [], 'val_acc': [], 'val_acc_scores': [], 'val_prec_scores': [], 'val_rec_scores': [],
    #       'val_ham_loss': []}

    overall_metrics = {'training': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                                    'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                                    'samples/f1': [], 'hamming_loss': [], 'total_loss': []},
                       'validating': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                                      'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                                      'samples/f1': [], 'hamming_loss': [], 'total_loss': []}
                       }

    for ep in range(epochs):
        print(f'Training epoch {ep} ..... ')

        epoch_metrics = train_epoch(model, train_loader, optimizer=optimizer, lr=lr, loss_fn=loss_fn,
                                    device=device)
        val_metrics = validate(model, validation_dataloader, loss_fn=loss_fn, device=device)

        # print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        # res['train_loss'].append(tl)
        # res['train_acc'].append(ta)
        # res['train_acc_scores'].append(a_s)
        # res['train_prec_scores'].append(p_s)
        # res['train_rec_scores'].append(r_s)
        # res['train_ham_loss'].append(h_l)
        #
        # res['val_loss'].append(vl)
        # res['val_acc'].append(va)
        # res['val_acc_scores'].append(va_s)
        # res['val_prec_scores'].append(vp_s)
        # res['val_rec_scores'].append(vp_s)
        # res['val_ham_loss'].append(vh_l)

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


def show_4_image_in_batch(images_batch, predicted_labels):
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

    fig, axs = plt.subplots(1, 4)
    for i in range(4):
        img = F.to_pil_image(images_batch[i])
        axs[i].imshow(img)
        ids = num_tags[predicted_labels[i, :] == 1.0]
        axs[i].set_title(f'#{i}:\n {ids}')
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.tight_layout()
    plt.show()
