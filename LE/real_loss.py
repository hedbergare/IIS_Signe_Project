


def real_loss(Y_exist, Y_real):
    loss = 0
    for i in range(len(Y_exist)):
        if Y_exist[i] <0:
            if Y_real[i] == 1:
                loss += 1
        else:
            if Y_real[i] == -1:
                loss += 1
    return loss





