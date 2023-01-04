import torch


def distance_loss(Y_pred, Y):
    loss = 0
    for i in range(len(Y)):
        if Y[i][0] > -0.9983:
            loss += torch.sqrt((Y[i][0]-Y_pred[i][0]) **
                               2+(Y[i][1]-Y_pred[i][1])**2)
    return loss


def binary_loss(Y_pred, Y):
    loss = 0
    for i in range(len(Y_pred)):
        loss += (Y_pred[i][0]-Y[i])**2
    return loss
