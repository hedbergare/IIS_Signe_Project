from loss_function import distance_loss


def real_loss(Y_cord, Y_exist, Y_real):
    for i in range(len(Y_cord)):
        if Y_exist[i][0] < Y_exist[i][1]:
            Y_cord[i][0] = 0
            Y_cord[i][1] = 0
    return distance_loss(Y_cord, Y_real)
