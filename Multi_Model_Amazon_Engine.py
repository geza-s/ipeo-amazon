from tqdm import tqdm
import torch.nn as nn
from torch import no_grad
from torch import max as torch_max

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss


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
    :return: 5 numpy arrays for : total loss array, all accuracy array, precision score array,recall score array
    """

    sig = nn.Sigmoid()
    smax = nn.Softmax()

    ground_model.eval()
    cloud_model.eval()

    accs, acc_scores, prec_scores, rec_scores, tot_loss, ham_loss = [], [], [], [], [], []
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
            tot_loss.append(loss.cpu().detach().item())

            # Prediction of this batch and appending to all accuracies of this epoch
            predicted_ground = (sig(out_ground) > 0.5).float().cpu().detach().numpy()
            predicted_cloud = out_clouds.cpu().detach().numpy()
            predicted_cloud = (predicted_cloud == predicted_cloud.max(axis=1, keepdims=True))

            predicted = np.hstack((predicted_ground, predicted_cloud))
            ground_truth = np.hstack((ground_target.cpu().detach().numpy(), cloud_target.cpu().detach().numpy()))

            # save metrics
            accs.append(np.mean(np.array(predicted == ground_truth), axis=0).tolist())
            acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
            prec_scores.append(precision_score(ground_truth.flatten(), predicted.flatten()))
            rec_scores.append(recall_score(ground_truth.flatten(), predicted.flatten()))
            ham_loss.append(hamming_loss(ground_truth.flatten(), predicted.flatten()))

    return tot_loss, accs, acc_scores, prec_scores, rec_scores, ham_loss


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

    accs, acc_scores, prec_scores, rec_scores, tot_loss, ham_loss = [], [], [], [], [], []

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
        loss = loss_ground + loss_clouds                    # HERE TO CHECK AGAIN WHAT IS TOTAL LOSS !!!
        tot_loss.append(loss.cpu().detach().item())

        # Prediction of this batch and appending to all accuarcies of this epoch
        predicted_ground = (sig(out_ground) > 0.5).float().cpu().detach().numpy()
        predicted_cloud = out_clouds.cpu().detach().numpy()
        predicted_cloud = (predicted_cloud == predicted_cloud.max(axis=1, keepdims=True))

        predicted = np.hstack((predicted_ground, predicted_cloud))
        ground_truth = np.hstack((ground_target.cpu().detach().numpy(), cloud_target.cpu().detach().numpy()))

        # save all the metrics
        accs.append(np.mean(np.array(predicted == ground_truth), axis=0).tolist())
        acc_scores.append(accuracy_score(ground_truth.flatten(), predicted.flatten()))
        prec_scores.append(precision_score(ground_truth.flatten(), predicted.flatten()))
        rec_scores.append(recall_score(ground_truth.flatten(), predicted.flatten()))
        ham_loss.append(hamming_loss(ground_truth.flatten(), predicted.flatten()))

        if i_batch == 0:
            print(image_batch.size())
            print(np.shape(predicted), np.shape(ground_truth))
            print(
                f"Predicted : {predicted}, calculated accuracy score: {np.mean(acc_scores)}, prediction score : {np.mean(prec_scores)}, recall score: {np.mean(rec_scores)}")
            break
        if i_batch % 20 == 0:  # print every ... mini-batches the mean loss up to now
            print(
                f"Loss : {np.mean(tot_loss)}, calculated accuracy score: {np.mean(acc_scores)}, prediction score : {np.mean(prec_scores)}, recall score: {np.mean(rec_scores)}")

    return tot_loss, accs, acc_scores, prec_scores, rec_scores, ham_loss


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

    res = {'train_loss': [], 'train_acc': [], 'train_acc_scores': [], 'train_prec_scores': [], 'train_rec_scores': [],
           'train_ham_loss': [],
           'val_loss': [], 'val_acc': [], 'val_acc_scores': [], 'val_prec_scores': [], 'val_rec_scores': [],
           'val_ham_loss': []}
    for ep in range(epochs):
        tl, ta, a_s, p_s, r_s, h_l = train_epoch_dual(cloud_model, ground_model, train_loader, device=device,
                                                      ground_optimizer=ground_optimizer,
                                                      cloud_optimizer=cloud_optimizer, lr=lr,
                                                      ground_loss_fn=ground_loss_fn,
                                                      cloud_loss_fn=cloud_loss_fn)
        vl, va, va_s, vp_s, vr_s, vh_l = validate_dual(ground_model, cloud_model, validation_dataloader,
                                                       ground_loss_fn=ground_loss_fn,
                                                       cloud_loss_fn=cloud_loss_fn, device=device)

        # print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['train_acc_scores'].append(a_s)
        res['train_prec_scores'].append(p_s)
        res['train_rec_scores'].append(r_s)
        res['train_ham_loss'].append(h_l)

        res['val_loss'].append(vl)
        res['val_acc'].append(va)
        res['val_acc_scores'].append(va_s)
        res['val_prec_scores'].append(vp_s)
        res['val_rec_scores'].append(vp_s)
        res['val_ham_loss'].append(vh_l)
    return res


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
    accuracy = (np.array((predicted == ground_truth)).astype(np.float64).mean())

    return loss.cpu().detach().numpy(), accuracy, predictions

#%%
