from tqdm import tqdm
import torch.nn as nn
from torch import no_grad
from torch import max as torch_max
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn import metrics

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


def validate_dual(ground_model, cloud_model, dataloader, device, ground_loss_fn=nn.BCEWithLogitsLoss(),
                  cloud_loss_fn=nn.BCELoss()):
    """
    Function running validation loop accros the validation dataloader on dual model
    :param ground_model: ground label model to validate
    :param cloud_model: cloud detection model to validate
    :param dataloader: validation dataloader
    :param device: device : "cuda" or "cpu"
    :param ground_loss_fn: default = BCEWithLogitsLoss() for multiple ground-label classification
    :param cloud_loss_fn: default = BCELoss for 3 cloud classes classification
    :return: ...
    """

    sig = nn.Sigmoid() #sigmoid function needed for estimated prediction

    ground_model.eval()
    cloud_model.eval()

    overall_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                     'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                     'samples/f1': [], 'hamming_loss': [], 'total_loss':[]}

    print('Validating')

    with no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader)):
            image_batch = sample_batch['image'].to(device)
            cloud_target = sample_batch['cloud_target'].to(device)
            ground_target = sample_batch['ground_target'].to(device)

            # Prediction from model
            out_clouds = cloud_model(image_batch)
            out_ground = ground_model(image_batch)

            # Loss function -> sigmoid included in BCEWithLogistLoss
            loss_clouds = cloud_loss_fn(out_clouds, cloud_target)
            loss_ground = ground_loss_fn(out_ground, ground_target)

            # Metrics
            # All the losses for this epoch
            loss = loss_ground + loss_clouds  # HERE TO CHECK AGAIN WHAT IS TOTAL LOSS !!!
            #tot_loss.append(loss.cpu().detach().item())

            # Prediction of this batch and appending to all accuracies of this epoch
            predicted_ground = (sig(out_ground) > 0.5).float().cpu().detach().numpy()
            predicted_cloud = out_clouds.cpu().detach().numpy()
            predicted_cloud = (predicted_cloud == predicted_cloud.max(axis=1, keepdims=True))

            predicted = np.hstack((predicted_ground, predicted_cloud))
            ground_truth = np.hstack((ground_target.cpu().detach().numpy(), cloud_target.cpu().detach().numpy()))

            # save metrics
            batch_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().item())

            # accs.append(np.mean(np.array(predicted == ground_truth), axis=0).tolist())
            # acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
            # prec_scores.append(precision_score(ground_truth.flatten(), predicted.flatten()))
            # rec_scores.append(recall_score(ground_truth.flatten(), predicted.flatten()))
            # ham_loss.append(hamming_loss(ground_truth.flatten(), predicted.flatten()))

            # Append metrics to the overall epoch metrics measures
            append_metrics(overall_metrics, batch_metrics)

    return overall_metrics


def train_epoch_dual(cloud_model, ground_model, dataloader, device, lr=0.01, ground_optimizer=None,
                     cloud_optimizer=None, ground_loss_fn=nn.BCEWithLogitsLoss(), cloud_loss_fn=nn.BCELoss()):
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
    clouds_optimizer = cloud_optimizer or torch.optim.Adam(cloud_model.parameters(), lr=lr)
    ground_optimizer = ground_optimizer or torch.optim.Adam(ground_model.parameters(), lr=lr)

    cloud_model.train()
    ground_model.train()

    epoch_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                     'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                     'samples/f1': [], 'hamming_loss': [], 'total_loss':[]}

    print('Training')
    for i_batch, sample_batch in tqdm(enumerate(dataloader)):
        # get the inputs.
        image_batch = sample_batch['image'].to(device)
        cloud_target = sample_batch['cloud_target'].to(device)
        ground_target = sample_batch['ground_target'].to(device)

        # zero the parameter gradients
        clouds_optimizer.zero_grad()
        ground_optimizer.zero_grad()

        # Prediction from model
        out_clouds = cloud_model(image_batch)
        out_ground = ground_model(image_batch)

        # Loss function -> sigmoid included in BCEWithLogistLoss
        loss_clouds = cloud_loss_fn(out_clouds, cloud_target)
        loss_ground = ground_loss_fn(out_ground, ground_target)

        # backpropagation
        loss_clouds.backward()
        loss_ground.backward()

        # update optimizer parameters
        clouds_optimizer.step()
        ground_optimizer.step()

        # Metrics
        # All the losses for this epoch
        loss = loss_ground + loss_clouds  # HERE TO CHECK AGAIN WHAT IS TOTAL LOSS !!!

        #tot_loss.append(loss.cpu().detach().item())

        # Prediction of this batch and appending to all accuarcies of this epoch
        predicted_ground = (sig(out_ground) > 0.5).float().cpu().detach().numpy()
        predicted_cloud = out_clouds.cpu().detach().numpy()
        predicted_cloud = (predicted_cloud == predicted_cloud.max(axis=1, keepdims=True))

        predicted = np.hstack((predicted_ground, predicted_cloud))
        ground_truth = np.hstack((ground_target.cpu().detach().numpy(), cloud_target.cpu().detach().numpy()))

        # get the metrics
        batch_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().item())

        # accs.append(np.mean(np.array(predicted == ground_truth), axis=0).tolist())
        # acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
        # prec_scores.append(precision_score(ground_truth.flatten(), predicted.flatten()))
        # rec_scores.append(recall_score(ground_truth.flatten(), predicted.flatten()))
        # ham_loss.append(hamming_loss(ground_truth.flatten(), predicted.flatten()))

        # Append metrics to the overall epoch metrics measures
        append_metrics(epoch_metrics, batch_metrics)

        if i_batch == 0:
            print(image_batch.size())
            print(np.shape(predicted), np.shape(ground_truth))
            print("iter:{:3d} training:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                              batch_metrics['samples/f1']))
            continue

        if i_batch % 20 == 0:  # print every ... mini-batches the mean loss up to now
            print("iter:{:3d} training:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                              batch_metrics['samples/f1']))

    return epoch_metrics


def train_dual(ground_model, cloud_model, train_loader, validation_dataloader, device, ground_optimizer=None,
               cloud_optimizer=None, lr=0.01, epochs=2, ground_loss_fn=None, cloud_loss_fn=None):
    """
    :param ground_model: model for ground label detection
    :param cloud_model: model for cloud detection
    :param train_loader: train dataloader
    :param validation_dataloader : dataloader associated to validation
    :param device: device for training
    :param ground_optimizer: optimizer per default is Adam
    :param cloud_optimizer: optimizer per default is Adam
    :param lr: default 0.01
    :param epochs: default 2
    :param ground_loss_fn: default BCEWithLogitsLoss (sigmoid + Cross Entropy)
    :param cloud_loss_fn: default BCELoss (after softmax)
    :return: results as a dictionary with 'train_loss', 'train_acc', 'train_acc_scores', 'train_prec_scores',
     'train_rec_scores', 'train_ham_loss', 'val_loss', 'val_acc', 'val_acc_scores', 'val_prec_scores',
     'val_rec_scores', 'val_ham_loss'
    """

    ground_optimizer = ground_optimizer or torch.optim.Adam(ground_model.parameters(), lr=lr)
    cloud_optimizer = cloud_optimizer or torch.optim.Adam(cloud_model.parameters(), lr=lr)

    overall_metrics = {'training': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                                    'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                                    'samples/f1': [], 'hamming_loss': [], 'total_loss':[]},
                       'validating': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                                      'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                                      'samples/f1': [], 'hamming_loss': [], 'total_loss':[]}
                       }

    for ep in range(epochs):
        epoch_metrics = train_epoch_dual(cloud_model, ground_model, train_loader, device=device,
                                                      ground_optimizer=ground_optimizer,
                                                      cloud_optimizer=cloud_optimizer, lr=lr,
                                                      ground_loss_fn=ground_loss_fn,
                                                      cloud_loss_fn=cloud_loss_fn)
        val_metrics = validate_dual(ground_model, cloud_model, validation_dataloader,
                                                       ground_loss_fn=ground_loss_fn,
                                                       cloud_loss_fn=cloud_loss_fn, device=device)

        # print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")

        # Append all the metrics
        append_mean_metrics(overall_metrics['training'], epoch_metrics)
        append_mean_metrics(overall_metrics['validating'], val_metrics)

    return overall_metrics


def batch_prediction_dual(batch, ground_model, cloud_model, device="cuda", ground_criterion=nn.BCEWithLogitsLoss(),
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
    #accuracy = (np.array((predicted == ground_truth)).astype(np.float64).mean())

    bach_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().numpy())

    return bach_metrics, predictions
