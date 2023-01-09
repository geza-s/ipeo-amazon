import numpy as np
from datetime import date
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch.nn as nn
from torch import no_grad
from torch import save as torch_save
from torch import max as torch_max
from torch.optim import Adam

from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, confusion_matrix


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


def validate_solo(model_type, model, dataloader, device, loss_fn=None):
    """
    Function running validation loop accros the validation dataloader on dual model
    :param model_type: "ground_model" or "cloud_model"
    :param model: cloud detection model to validate
    :param dataloader: validation dataloader
    :param device: device : "cuda" or "cpu"
    :param loss_fn: no default !
    :return: ...
    """

    sig = nn.Sigmoid()  # sigmoid function needed for estimated prediction

    model.eval()

    if model_type == 'ground_model':
        print("Validating ground model")
        val_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                       'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                       'samples/f1': [], 'hamming_loss': [], 'total_loss': []}
    elif model_type == 'cloud_model':
        print('Validating cloud model')
        val_metrics = {'accuracy': [], 'total_loss': []}
    else:
        print("Couldn't detect model type to train... Should be either ground_model or cloud_model..")
        return False

    with no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader)):
            image_batch = sample_batch['image'].to(device)
            cloud_target = sample_batch['cloud_target'].to(device)
            ground_target = sample_batch['ground_target'].to(device)

            if model_type == 'ground_model':

                # Prediction from model
                out_ground = model(image_batch)

                # Loss function -> sigmoid included in BCEWithLogistLoss
                loss_ground = loss_fn(out_ground, ground_target)

                # Prediction of this batch and appending to all accuracies of this epoch
                predicted_ground = (sig(out_ground) > 0.5).float().cpu().detach().numpy()

                ground_batch_metrics = calculate_metrics(predicted_ground, ground_target.cpu().detach().numpy(),
                                                         loss_ground.cpu().detach().item())
                append_metrics(val_metrics, ground_batch_metrics)

            elif model_type == 'cloud_model':

                # Prediction from model
                out_clouds = model(image_batch)

                # Loss function -> sigmoid included in BCEWithLogistLoss
                loss_clouds = loss_fn(out_clouds, cloud_target)

                # Prediction of this batch and appending to all accuracies of this epoch
                predicted_cloud = out_clouds.cpu().detach().numpy()
                predicted_cloud = (predicted_cloud == predicted_cloud.max(axis=1, keepdims=True))

                # calculate metrics:
                cloud_batch_metrics = {'accuracy': accuracy_score(predicted_cloud, cloud_target.cpu().detach().numpy()),
                                       'total_loss': loss_clouds.cpu().item()}
                append_metrics(val_metrics, cloud_batch_metrics)

    return val_metrics


def train_epoch_solo(model_type, model, dataloader, device, lr=0.01, optimizer=None, loss_fn=None):
    """
    Loop for 1 epoch for a dual model approach
    :param ground_model: model used for ground prediction
    :param cloud_model: model used for cloud prediction
    :param dataloader: dataloader over the amazon space dataset
    :param device: device on which train, 'cuda' or 'cpu'
    :param lr: learning rate, here default = 0.01
    :param ground_optimizer: Here if None given then using Adam
    :param cloud_optimizer: Here if None given then using Adam
    :param ground_loss_fn: Loss function default is BCEWithLogitsLoss (sigmoid + cross-entropy)
    :param cloud_loss_fn: loss function associated with the cloud prediction model
    :return: [tot_loss, accs, acc_scores, prec_scores, rec_scores, ham_loss]
    """
    sig = nn.Sigmoid()
    model.train()

    optimizer = optimizer or Adam(model.parameters(), lr=lr)
    print(optimizer)

    if model_type == 'ground_model':
        print("Training a ground model")
        target_name = "ground_target"
        epoch_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                         'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                         'samples/f1': [], 'hamming_loss': [], 'total_loss': []}
    elif model_type == 'cloud_model':
        print("Training a cloud model")
        target_name = "cloud_target"
        epoch_metrics = {'accuracy': [], 'total_loss': []}
    else:
        print("Couldn't detect model type to train... Should be either ground_model or cloud_model..")
        return False

    for i_batch, sample_batch in tqdm(enumerate(dataloader)):
        # get the inputs.
        image_batch = sample_batch['image'].to(device)
        target = sample_batch[target_name].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # Prediction from model
        out = model(image_batch)
        # Loss function -> sigmoid included in BCEWithLogistLoss
        loss = loss_fn(out, target)
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()

        if model_type == 'ground_model':
            # Prediction of this batch and appending to all accuarcies of this epoch
            predicted = (sig(out) > 0.5).float().cpu().detach().numpy()

            # get the metrics
            batch_metrics = calculate_metrics(predicted, target.cpu().detach().numpy(),
                                              loss.cpu().detach().item())
            if i_batch == 0:
                print(image_batch.size())
                print(np.shape(predicted), np.shape(target))
                print("iter:{:3d} training:"
                      "micro f1: {:.3f}"
                      "macro f1: {:.3f} "
                      "samples f1: {:.3f}"
                      "loss {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                           batch_metrics['samples/f1'], batch_metrics['total_loss']))
                show_4_image_in_batch_solo(model_type, image_batch, predicted,
                                           ground_truth=target.cpu().detach().numpy())
                continue

            if i_batch % 30 == 0:  # print every ... mini-batches the mean loss up to now
                print("Ground metrics:"
                      "micro f1: {:.3f}"
                      "macro f1: {:.3f} "
                      "samples f1: {:.3f}"
                      "loss {:.3f}".format(batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                           batch_metrics['samples/f1'], batch_metrics['total_loss']))

        elif model_type == 'cloud_model':
            # Prediction of this batch and appending to all accuarcies of this epoch
            predicted = out.cpu().detach().numpy()
            predicted = (predicted == predicted.max(axis=1, keepdims=True))

            # get the metrics
            batch_metrics = {'accuracy': accuracy_score(predicted, target.cpu().detach().numpy()),
                             'total_loss': loss.cpu().item()}
            if i_batch == 0:
                print(image_batch.size())
                print(np.shape(predicted), np.shape(target))
                print(f"Cloud metrics : {batch_metrics['accuracy']} and loss:{batch_metrics['total_loss']}")
                show_4_image_in_batch_solo("cloud_model", image_batch, predicted,
                                           ground_truth=target.cpu().detach().numpy())
                continue

            if i_batch % 30 == 0:  # print every ... mini-batches the mean loss up to now
                print(f"Cloud metrics : acc: {batch_metrics['accuracy']} and loss:{batch_metrics['total_loss']}")
        else:
            print(f"Couldn't train model type {model_type}")
            return False

        if i_batch % 100 == 0:
            show_4_image_in_batch_solo(model_type, image_batch, predicted, ground_truth=target.cpu().detach().numpy())

        # Append metrics to the overall epoch metrics measures
        append_metrics(epoch_metrics, batch_metrics)

    return epoch_metrics


def train_solo(model_type, model, train_loader, validation_dataloader, device, optimizer=None, lr=0.01, epochs=2,
               loss_fn=None):
    """

    :param model_type:
    :param cloud_model:
    :param train_loader:
    :param validation_dataloader:
    :param device:
    :param optimizer:
    :param lr:
    :param epochs:
    :param loss_fn:
    :return:
    """

    optimizer = optimizer or Adam(model.parameters(), lr=lr)

    if model_type == 'ground_model':
        overall_metrics = {'training': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [],
                                        'macro/precision': [], 'macro/recall': [], 'macro/f1': [],
                                        'samples/precision': [], 'samples/recall': [], 'samples/f1': [],
                                        'hamming_loss': [], 'total_loss': []},
                           'validating': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [],
                                          'macro/precision': [], 'macro/recall': [], 'macro/f1': [],
                                          'samples/precision': [], 'samples/recall': [], 'samples/f1': [],
                                          'hamming_loss': [], 'total_loss': []}
                           }
    elif model_type == 'cloud_model':
        overall_metrics = {'training': {'accuracy': [], 'total_loss': []},
                           'validating': {'accuracy': [], 'total_loss': []}
                           }
    else:
        print(f"Couldn't fin model type {model_type}")

    min_loss = 1000

    for ep in range(epochs):
        print(f"Training on epoch {ep}............")

        epoch_metrics = train_epoch_solo(model_type, model, train_loader,
                                         device=device,
                                         optimizer=optimizer, lr=lr, loss_fn=loss_fn)
        val_metrics = validate_solo(model_type, model, validation_dataloader, device=device, loss_fn=loss_fn)

        # check if best performance, save if yes
        if np.mean(val_metrics['total_loss']) < min_loss:
            min_loss = np.mean(val_metrics['total_loss'])
            if model_type == 'ground_model':
                model_save_name = f"groundmodel_NewClassifier_{epochs}epochs_{date.today()}.pth"
            elif model_type == 'cloud_model':
                model_save_name = f"cloudmodel_NewClassifier_{epochs}epochs_{date.today()}.pth"
            else:
                model_save_name = f"{model_type}_error.pth"

            model.eval()
            torch_save(model.state_dict(), model_save_name)
            print(f"Saved PyTorch Model State to {model_save_name}")

        # Append all the metrics
        append_mean_metrics(overall_metrics['training'], epoch_metrics)
        append_mean_metrics(overall_metrics['validating'], val_metrics)

    print(".........  ENDED TRAINING !!")
    return overall_metrics


def batch_prediction_dual(batch, ground_model, cloud_model, device, ground_criterion=nn.BCEWithLogitsLoss(),
                          cloud_criterion=nn.BCELoss()):
    """
    Predict the values from model for batch given. Function for the testing of the model.
    :param batch: batch composed of 'image' and 'labels'
    :param ground_model: model trained for ground detection
    :param cloud_model: model trained for cloud detection
    :param device: on which device run the thing
    :param ground_criterion: for ground loss value
    :param cloud_criterion: for cloud loss value
    :return: loss,accuracy
    """
    ground_model.eval()
    cloud_model.eval()

    sig = nn.Sigmoid()

    # Retrieve image and label from the batch and
    x = batch['image'].to(device)
    y_ground = batch['ground_target'].to(device)
    y_cloud = batch['cloud_target'].to(device)

    # move model and code to GPU
    ground_model = ground_model.to(device)
    cloud_model = cloud_model.to(device)

    # Forward pass
    y_hat_ground = ground_model(x)
    y_hat_cloud = cloud_model(x)

    # Loss calculation (only for statistics)
    ground_loss = ground_criterion(y_hat_ground, y_ground)
    cloud_loss = cloud_criterion(y_hat_cloud, y_cloud)

    loss = ground_loss + cloud_loss  ## HERE ALSO TO MODIFY ACCORDING TO LOSS

    # THe predictions
    ground_predicted = (sig(y_hat_ground) > 0.5).float().cpu().detach().numpy()
    _, cloud_predicted = torch_max(y_hat_cloud, 1)

    predicted = np.hstack((ground_predicted, cloud_predicted.cpu().detach().numpy))

    # The targeted values
    ground_target = y_ground.cpu().detach().numpy()
    cloud_target = y_cloud.cpu().detach().numpy()
    ground_truth = np.hstack((ground_target, cloud_target))

    # The according metrics
    predictions = np.array((predicted == ground_truth), dtype=np.float64).mean(axis=0)
    # accuracy = (np.array((predicted == ground_truth)).astype(np.float64).mean())

    bach_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().numpy())

    return bach_metrics, predictions


def show_4_image_in_batch_solo(model_type, images_batch, predicted_labels, ground_truth):
    """
    Shows 4 first images from the batch of the Amazon Dataset
    :param model_type: "combined", "cloud_model", "ground_model"
    :param sample_batched: mini-batch of dataloader of Amazon Dataset. Dictionary with 'image', 'labels'
    :param tags: All the unique labels
    :param ground_truth: The ground truth vector for labels
    :return:
    """
    if model_type == 'combined':
        tags = ['haze', 'primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
                'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down',
                'clear', 'cloudy', 'partly_cloudy']
    elif model_type == 'ground_model':
        tags = ['haze', 'primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
                'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
    elif model_type == 'cloud_model':
        tags = ['clear', 'cloudy', 'partly_cloudy']
    else :
        print(f"Model type {model_type} not recognised for image showing")

    num_tags = np.arange(start=0, stop=len(tags))

    fig, axs = plt.subplots(1, 4, sharey=True)
    for i in range(4):
        img = to_pil_image(images_batch[i])
        axs[i].imshow(img)
        ids = num_tags[predicted_labels[i, :] == 1.0]
        ids_truth = num_tags[ground_truth[i, :] == 1]
        names = [tags[ix] for ix in ids]
        names_truth = [tags[i] for i in ids_truth]
        axs[i].set_title(f'#{i}:\n pred: {names} \n truth: {names_truth}', {'fontsize': 12})
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.tight_layout()
    plt.show()
