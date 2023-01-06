from model_no_cord import UNetModelFlatten
from ny_ladda_data import ladda
from Unet import UNetModelDNNST
from loss_function import distance_loss
import dsntnn
import torch

def crossval(batch_size,flatten=True):
    print("kör")
    device = torch.device('mps')
    deeps=[2,3,4,5,6]
    mini_batch_size = 5
    valdeep=[]
    for deep in deeps:
        validationround=1
        while (validationround<6):
            
            print("currentfold, " + str(validationround) )
            if (flatten):
                model = UNetModelFlatten(deep, 3, 20, 64).to(device)
            else:
                model = UNetModelDNNST(deep, 3, 20, 64).to(device)
            optim = torch.optim.SGD(model.parameters(), lr=0.00000001, momentum=0.9)
                
            
            model.train()
            for batch in range(batch_size):
                print("starting batch, "  )
                loss = 0
                train_data = ladda(typ='träning', seed=2, validation_round=validationround, data_divided_into=batch_size, data_pass=batch)
                for i,frame in enumerate(train_data):

                    image=frame[0].to(device)
                    Y_pred=frame[1].to(device)

                    if(flatten):
                        pred = model(image)
                        loss += distance_loss(pred, Y_pred)

                    else:
                        pred,heatmaps = model(image)
                        euc_losses = dsntnn.euclidean_losses(pred, Y_pred)
                        # Per-location regularization losses
                        reg_losses = dsntnn.js_reg_losses(heatmaps, Y_pred, sigma_t=1.0)
                            # Combine losses into an overall loss
                        loss+= dsntnn.average_loss(euc_losses + reg_losses)

                    if i % (mini_batch_size) == (mini_batch_size-1) or i == (len(train_data)-1):
                        print(i,"Loss: ", loss/mini_batch_size)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        loss = 0
                    del image
                    del Y_pred
        val_loss = 0
        for batch in batch_size//5:   
            loss = 0      
            val_data=ladda(typ="validering", seed=2, validation_round=validationround, data_divided_into=batch_size, data_pass=batch)
            for frame in val_data:
                image = frame[0].to(device)
                true_pred = frame[1].to(device)
                model_pred = model(image)
                loss += distance_loss(model_pred, true_pred)
                del image 
                del true_pred
            val_loss += loss/len(batch)
        val_loss = val_loss/5
        valdeep.append(val_loss)
    return valdeep

print(crossval(40))
            



