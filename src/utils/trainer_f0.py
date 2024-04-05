import torch
import numpy as np
import os
from tqdm import tqdm


def train(model, optimizer, data_loader, device):
    
    model.train()
    
    ccounter = 0.0
    total_f0_loss = 0.0

    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        outputs = model(batch, "train")
        loss = outputs["train_f0_loss"]
        loss.backward()
        optimizer.step()
        total_f0_loss += loss.detach().cpu().numpy()
        ccounter += 1
    loss_f0 = float(total_f0_loss) / ccounter
    
    return loss_f0
 
   
def val(model, data_loader, deivce):
    model.eval()
    
    total = {"f0_loss":0}
    ccounter = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outputs = model(batch, "val")
            total["f0_loss"] += outputs["val_f0_loss"].detach().cpu().numpy()
            ccounter += 1
        f0_loss = float(total["f0_loss"]) / ccounter

    return f0_loss


def trainer(num_epochs, model, loader_dict, optimizer, device, outdir):

    best_val_f0_loss = 1000000000
    for epoch in range(num_epochs):
        print("Epoch:{}".format(epoch+1))
        train_f0_loss = train(model, optimizer, loader_dict['train'], device)
        val_f0_loss = val(model, loader_dict['val'], device)

        print('------------------------------------')
        print("Train F0 MSE loss: {}".format(train_f0_loss))
        print('------------------------------------')
        print("Val F0 MSE loss: {}".format(val_f0_loss))
        print('------------------------------------')
        if best_val_f0_loss > val_f0_loss:
            best_val_f0_loss = val_f0_loss
            torch.save(model.state_dict(), os.path.join(outdir, "best_val_loss_model.pth"))
        
            
def tester(model, loader_dict, device):
    test_f0_loss = val(model, loader_dict['test'], device)
    print('------------------------------------')
    print("Test F0 MSE loss: {}".format(test_f0_loss))
    print('------------------------------------')