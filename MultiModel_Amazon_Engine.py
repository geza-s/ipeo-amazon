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
from sklearn.metrics import precision_recall_fscore_support


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
    p_micro = precision_recall_fscore_support(y_true=target, y_pred=pred, average='micro', zero_division=0)
    p_macro = precision_recall_fscore_support(y_true=target, y_pred=pred, average='macro', zero_division=0)
    p_samples = precision_recall_fscore_support(y_true=target, y_pred=pred, average='samples', zero_division=0)

    results = {'micro/precision': p_micro[0], 'micro/recall': p_micro[1], 'micro/f1': p_micro[2],
               'macro/precision': p_macro[0], 'macro/recall': p_macro[1], 'macro/f1': p_macro[2],
               'samples/precision': p_samples[0], 'samples/recall': p_samples[1], 'samples/f1': p_samples[2],
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

    epoch_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                     'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                     'samples/f1': [], 'hamming_loss': [], 'total_loss': []}

    if model_type == 'ground_model':
        print("Training the ground model")
        target_name = "ground_target"
    elif model_type == 'cloud_model':
        print("Training the cloud model")
        target_name = "cloud_target"
    else:
        print("Couldn't detect model type to train... Should be either ground_model or cloud_model..")
        return False

    for i_batch, sample_batch in tqdm(enumerate(dataloader)):
        # Get the batch's image to train on
        image_batch = sample_batch['image'].to(device)

        # Get the batch's targeted labels ("ground truth")
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
            # First apply sigmoid and threshold of 0.5 to get prediction of ground model
            predicted = (sig(out) > 0.5).float().cpu().detach().numpy()
        elif model_type == 'cloud_model':
            # Cloud model has an included softmax layer, take only the highest value.
            predicted = out.cpu().detach().numpy()
            predicted = (predicted == predicted.max(axis=1, keepdims=True))
        else:
            print(f"Couldn't train model type {model_type}")
            return False

        # get the metrics
        batch_metrics = calculate_metrics(predicted, target.cpu().detach().numpy(), loss.cpu().detach().item())

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

        if i_batch % 500 == 0:  # print every ... mini-batches the mean loss up to now
            print("Model metrics:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}"
                  "loss {:.3f}".format(batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                       batch_metrics['samples/f1'], batch_metrics['total_loss']))

        if i_batch % 1000 == 0:
            show_4_image_in_batch_solo(model_type, image_batch, predicted, ground_truth=target.cpu().detach().numpy())

        # Append metrics to the overall epoch metrics measures
        append_metrics(epoch_metrics, batch_metrics)

    return epoch_metrics

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

    val_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                   'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                   'samples/f1': [], 'hamming_loss': [], 'total_loss': []}

    if model_type == 'ground_model':
        print("Validating the ground model")
        target_name = "ground_target"
    elif model_type == 'cloud_model':
        print("Validating the cloud model")
        target_name = "cloud_target"
    else:
        print("Couldn't detect model type to validate... Should be either ground_model or cloud_model..")
        return False

    with no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader)):
            # Image from the batch to validate on
            image_batch = sample_batch['image'].to(device)

            # Targeted labels ("ground truth")
            target = sample_batch[target_name].to(device)

            # Prediction from model
            out = model(image_batch)

            # Loss function -> sigmoid included in BCEWithLogistLoss
            loss = loss_fn(out, target)

            if model_type == 'ground_model':
                # Prediction for ground model must pass through sigmoid and threshold=0.5
                predicted = (sig(out) > 0.5).float().cpu().detach().numpy()
            elif model_type == 'cloud_model':
                # Prediction for cloud model based on maximal value after the included softmax layer
                predicted = out.cpu().detach().numpy()
                predicted = (predicted == predicted.max(axis=1, keepdims=True))
            else:
                print(f"Error in model type {model_type}")
                return False

    #Calculate all the metrics
    batch_metrics = calculate_metrics(predicted, target.cpu().detach().numpy(), loss.cpu().detach().item())
    #Append all the metrics to the overall metrics for the validating phase
    append_metrics(val_metrics, batch_metrics)

    return val_metrics

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

    overall_metrics = {'training': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [],
                                    'macro/precision': [], 'macro/recall': [], 'macro/f1': [],
                                    'samples/precision': [], 'samples/recall': [], 'samples/f1': [],
                                    'hamming_loss': [], 'total_loss': []},
                       'validating': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [],
                                      'macro/precision': [], 'macro/recall': [], 'macro/f1': [],
                                      'samples/precision': [], 'samples/recall': [], 'samples/f1': [],
                                      'hamming_loss': [], 'total_loss': []}
                       }

    if model_type == 'ground_model':
        print("Working on a ground model...")
    elif model_type == 'cloud_model':
        print("Working on a cloud model...")
    else:
        print(f"Couldn't fin model type {model_type}")
        return False

    min_loss = 1000

    for ep in range(epochs):
        print(f"Training on epoch {ep}............")

        epoch_metrics = train_epoch_solo(model_type, model, train_loader, device=device, optimizer=optimizer, lr=lr,
                                         loss_fn=loss_fn)
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


def testing_multi(test_dataloader, model_ground, model_cloud, device="cpu",
                  criterion_gr=nn.BCEWithLogitsLoss(), criterion_cl=nn.BCELoss()):
    """
    Predict the values from model for test_dataloader given. Function for the testing of the model.
    :param test_dataloader:
    :param model: model of interest
    :param device: on which device run the thing
    :param tags: name of the classes
    :return: report, losses
    """

    tags_cl = test_dataloader.dataset.tags_cloud.keys()
    tags_gr = test_dataloader.dataset.tags_ground.keys()

    # store stats
    count = 0

    for batch in tqdm(test_dataloader):

        # run prediction_step
        results = batch_prediction_multi(batch, model_ground, model_cloud, device=device,
                                         criterion_gr=nn.BCEWithLogitsLoss(), criterion_cl=nn.BCELoss())

        # accuracies.append(accuracy)
        if count == 0:
            testing_results = {'ground': {'target': results['ground']['target'],
                                          'predicted': results['ground']['predicted'],
                                          'loss': results['ground']['loss']},
                               'cloud': {'target': results['cloud']['target'],
                                         'predicted': results['cloud']['predicted'],
                                         'loss': results['cloud']['loss']},
                               'total': {'target': results['total']['target'],
                                         'predicted': results['total']['predicted'],
                                         'loss': results['total']['loss']}
                               }
            count = 1

        else:
            testing_results = {
                'ground': {'target': np.vstack((testing_results['ground']['target'], results['ground']['target'])),
                           'predicted': np.vstack(
                               (testing_results['ground']['predicted'], results['ground']['predicted'])),
                           'loss': np.vstack((testing_results['ground']['loss'], results['ground']['loss']))},
                'cloud': {'target': np.vstack((testing_results['cloud']['target'], results['cloud']['target'])),
                          'predicted': np.vstack(
                              (testing_results['cloud']['predicted'], results['cloud']['predicted'])),
                          'loss': np.vstack((testing_results['cloud']['loss'], results['cloud']['loss']))},
                'total': {'target': np.vstack((testing_results['total']['target'], results['total']['target'])),
                          'predicted': np.vstack(
                              (testing_results['total']['predicted'], results['total']['predicted'])),
                          'loss': np.vstack((testing_results['total']['loss'], results['total']['loss']))}
                }

    return testing_results


def batch_prediction_multi(batch, model_ground, model_cloud, device="cpu",
                           criterion_gr=nn.BCEWithLogitsLoss(), criterion_cl=nn.BCELoss()):
    """
    Predict the values from ground and cloud models for batch given. Function for the testing of the model.
    :param batch: batch composed of 'image' and 'labels'
    :param model_ground: model for ground labels
    :param model_cloud: model for cloud labels
    :param device: on which device run the testing
    :return: 2 level dictionary of the results: {'ground', 'cloud' and 'total'} -> {'target', 'predicted' and 'loss'}
    """

    sig = nn.Sigmoid()

    model_ground.eval()
    model_cloud.eval()

    # Retrieve image and label from the batch
    x = batch['image'].to(device)
    y_gr = batch['ground_target'].to(device)
    y_cl = batch['cloud_target'].to(device)

    # Forward pass
    y_hat_gr = model_ground(x)
    y_hat_cl = model_cloud(x)

    # Loss calculation (only for statistics)
    loss_gr = criterion_gr(y_hat_gr, y_gr).cpu().detach().numpy()
    loss_cl = criterion_cl(y_hat_cl, y_cl).cpu().detach().numpy()

    loss_glob = loss_gr + loss_cl

    # Calculate accuracy for statistics
    predicted_gr = (sig(y_hat_gr) > 0.5).float().cpu().detach().numpy()
    ground_truth = y_gr.cpu().detach().numpy()
    #accuracy_gr = np.array((predicted_gr == ground_truth), dtype=np.float64).mean(axis=0)

    predicted_cl = y_hat_cl.cpu().detach().numpy()
    predicted_cl = (predicted_cl == predicted_cl.max(axis=1, keepdims=True))
    cloud_truth = y_cl.cpu().detach().numpy()
    #accuracy_cl = np.array((predicted_cl == cloud_truth), dtype=np.float64).mean(axis=0)

    results = {'ground': {'target': ground_truth,
                          'predicted': predicted_gr,
                          'loss': loss_gr},
               'cloud': {'target': cloud_truth,
                         'predicted': predicted_cl,
                         'loss': loss_cl},
               'total': {'target': np.hstack((ground_truth, cloud_truth)),
                         'predicted': np.hstack((predicted_gr, predicted_cl)),
                         'loss': loss_glob}
               }

    return results


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
    plt.grid(False)
    plt.show()
