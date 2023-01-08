from tqdm import tqdm
import torch.nn as nn
from torch import no_grad, save
from torch import max as torch_max
from torch.optim import Adam
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
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

    sig = nn.Sigmoid()  # sigmoid function needed for estimated prediction

    ground_model.eval()
    cloud_model.eval()

    overall_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                       'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                       'samples/f1': [], 'hamming_loss': [], 'total_loss': []}

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

            # Prediction of this batch and appending to all accuracies of this epoch
            predicted_ground = (sig(out_ground) > 0.5).float().cpu().detach().numpy()
            predicted_cloud = out_clouds.cpu().detach().numpy()
            predicted_cloud = (predicted_cloud == predicted_cloud.max(axis=1, keepdims=True))

            predicted = np.hstack((predicted_ground, predicted_cloud))
            ground_truth = np.hstack((ground_target.cpu().detach().numpy(), cloud_target.cpu().detach().numpy()))

            # save metrics
            batch_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().item())

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
    clouds_optimizer = cloud_optimizer or Adam(cloud_model.parameters(), lr=lr)
    ground_optimizer = ground_optimizer or Adam(ground_model.parameters(), lr=lr)

    cloud_model.train()
    ground_model.train()

    epoch_metrics = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                     'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                     'samples/f1': [], 'hamming_loss': [], 'total_loss': []}

    ground_metrics_epoch = {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                            'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                            'samples/f1': [], 'hamming_loss': [], 'total_loss': []}

    cloud_metrics_epoch = {'accuracy': [], 'total_loss': []}

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

        # tot_loss.append(loss.cpu().detach().item())

        # Prediction of this batch and appending to all accuarcies of this epoch
        predicted_ground = (sig(out_ground) > 0.5).float().cpu().detach().numpy()
        predicted_cloud = out_clouds.cpu().detach().numpy()
        predicted_cloud = (predicted_cloud == predicted_cloud.max(axis=1, keepdims=True))

        predicted = np.hstack((predicted_ground, predicted_cloud))
        ground_truth = np.hstack((ground_target.cpu().detach().numpy(), cloud_target.cpu().detach().numpy()))

        # get the metrics
        ground_batch_metrics = calculate_metrics(predicted_ground, ground_target.cpu().detach().numpy(),
                                                 loss_ground.cpu().detach().item())
        cloud_batch_metrics = {'accuracy': accuracy_score(predicted_cloud, cloud_target.cpu().detach().numpy()),
                               'total_loss': loss_clouds.cpu().item()}
        batch_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().item())

        # accs.append(np.mean(np.array(predicted == ground_truth), axis=0).tolist())
        # acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
        # prec_scores.append(precision_score(ground_truth.flatten(), predicted.flatten()))
        # rec_scores.append(recall_score(ground_truth.flatten(), predicted.flatten()))
        # ham_loss.append(hamming_loss(ground_truth.flatten(), predicted.flatten()))

        # Append metrics to the overall epoch metrics measures
        append_metrics(epoch_metrics, batch_metrics)
        append_metrics(ground_metrics_epoch, ground_batch_metrics)
        append_metrics(cloud_metrics_epoch, cloud_batch_metrics)

        if i_batch == 0:
            print(image_batch.size())
            print(np.shape(predicted), np.shape(ground_truth))
            print("iter:{:3d} training:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}"
                  "loss {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                              batch_metrics['samples/f1'], batch_metrics['total_loss']))
            print("Ground metrics: "
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f} "
                  "loss {:.3f}".format(ground_batch_metrics['micro/f1'], ground_batch_metrics['macro/f1'],
                                              ground_batch_metrics['samples/f1'], ground_batch_metrics['total_loss']))
            print(f"Cloud metrics : {cloud_batch_metrics['accuracy']} and loss:{cloud_batch_metrics['total_loss']}")
            show_4_image_in_batch(image_batch, predicted, ground_truth=ground_truth)
            continue

        if i_batch % 30 == 0:  # print every ... mini-batches the mean loss up to now
            print("iter:{:3d} training:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}"
                  "loss {:.3f}".format(i_batch, batch_metrics['micro/f1'], batch_metrics['macro/f1'],
                                       batch_metrics['samples/f1'], batch_metrics['total_loss']))
            print("Ground metrics:"
                  "micro f1: {:.3f}"
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}"
                  "loss {:.3f}".format(ground_batch_metrics['micro/f1'], ground_batch_metrics['macro/f1'],
                                       ground_batch_metrics['samples/f1'], ground_batch_metrics['total_loss']))
            print(f"Cloud metrics : {cloud_batch_metrics['accuracy']} and loss:{cloud_batch_metrics['total_loss']}")

        if i_batch % 100 == 0:
            show_4_image_in_batch(image_batch, predicted, ground_truth=ground_truth)

    return epoch_metrics, ground_metrics_epoch, cloud_metrics_epoch


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

    ground_optimizer = ground_optimizer or Adam(ground_model.parameters(), lr=lr)
    cloud_optimizer = cloud_optimizer or Adam(cloud_model.parameters(), lr=lr)

    overall_metrics = {'training': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                                    'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                                    'samples/f1': [], 'hamming_loss': [], 'total_loss': []},
                       'validating': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [], 'macro/precision': [],
                                      'macro/recall': [], 'macro/f1': [], 'samples/precision': [], 'samples/recall': [],
                                      'samples/f1': [], 'hamming_loss': [], 'total_loss': []},
                       'ground_model': {'micro/precision': [], 'micro/recall': [], 'micro/f1': [],
                                        'macro/precision': [],
                                        'macro/recall': [], 'macro/f1': [], 'samples/precision': [],
                                        'samples/recall': [],
                                        'samples/f1': [], 'hamming_loss': [], 'total_loss': []},
                       'cloud_model': {'accuracy': [], 'total_loss': []}
                       }

    min_loss = 1000

    for ep in range(epochs):
        print(f"Training on epoch {ep}............")

        epoch_metrics, ground_metrics, cloud_metrics = train_epoch_dual(cloud_model, ground_model, train_loader,
                                                                        device=device,
                                                                        ground_optimizer=ground_optimizer,
                                                                        cloud_optimizer=cloud_optimizer, lr=lr,
                                                                        ground_loss_fn=ground_loss_fn,
                                                                        cloud_loss_fn=cloud_loss_fn)
        val_metrics = validate_dual(ground_model, cloud_model, validation_dataloader,
                                    ground_loss_fn=ground_loss_fn,
                                    cloud_loss_fn=cloud_loss_fn, device=device)

        # check if best performance, save if yes
        if np.mean(epoch_metrics['total_loss']) < min_loss:
            min_loss = np.mean(epoch_metrics['total_loss'])
            grmodel_save_name = f"groundmodel_classifier_{epochs}epochs_{date.today()}.pth"
            clmodel_save_name = f"cloudmodel_classifier_{epochs}epochs_{date.today()}.pth"
            save(ground_model.state_dict(), grmodel_save_name)
            save(cloud_model.state_dict(), clmodel_save_name)
            print(f"Saved PyTorch Model State to {grmodel_save_name} and {clmodel_save_name}")

        # Append all the metrics
        append_mean_metrics(overall_metrics['training'], epoch_metrics)
        append_mean_metrics(overall_metrics['validating'], val_metrics)
        append_mean_metrics(overall_metrics['ground_model'], ground_metrics)
        append_mean_metrics(overall_metrics['cloud_model'], cloud_metrics)

    print(".........  ENDED TRAINING !!")
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
    # accuracy = (np.array((predicted == ground_truth)).astype(np.float64).mean())

    bach_metrics = calculate_metrics(predicted, ground_truth, loss.cpu().detach().numpy())

    return bach_metrics, predictions


def show_4_image_in_batch(images_batch, predicted_labels, ground_truth):
    """
    Shows 4 first images from the batch of the Amazon Dataset
    :param sample_batched: mini-batch of dataloader of Amazon Dataset. Dictionary with 'image', 'labels'
    :param tags: All the unique labels
    :param ground_truth: The ground truth vector for labels
    :return:
    """
    tags = ['haze', 'primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
            'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down',
            'clear', 'cloudy', 'partly_cloudy', 'none_clouds']
    num_tags = np.arange(start=0, stop=len(tags))

    # images_batch, labels = sample_batched['image'], sample_batched['labels']

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
