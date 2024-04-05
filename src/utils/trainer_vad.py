import torch
import numpy as np
import os
from tqdm import tqdm


def train(model, optimizer, data_loader, device):
    model.train()
    
    total = {"vad_loss":0, "vad_acc": 0, "vad_precision": 0, "vad_recall": 0, "vad_f1": 0}
    ccounter = 0.0
    
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        outputs = model(batch, "train")
        loss = outputs["train_loss"]
        loss.backward()
        optimizer.step()
        total["vad_loss"] += loss.detach().cpu().numpy()
        total["vad_acc"] += outputs["train_acc"]
        total["vad_precision"] += outputs["train_precision"]
        total["vad_recall"] += outputs["train_recall"]
        total["vad_f1"] += outputs["train_f1"]
        ccounter += 1
            
    vad_loss = float(total["vad_loss"]) / ccounter
    vad_acc = float(total["vad_acc"]) / ccounter
    vad_precision = float(total["vad_precision"]) / ccounter
    vad_recall = float(total["vad_recall"]) / ccounter
    vad_f1 = float(total["vad_f1"]) / ccounter
    
    return vad_loss, vad_acc, vad_precision, vad_recall, vad_f1
 
   
def val(model, data_loader, deivce):
    model.eval()
    
    total = {"vad_loss":0, "vad_acc": 0, "vad_precision": 0, "vad_recall": 0, "vad_f1": 0}
    ccounter = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outputs = model(batch, "val")
            total["vad_loss"] += outputs["val_loss"].detach().cpu().numpy()
            total["vad_acc"] += outputs["val_acc"]
            total["vad_precision"] += outputs["val_precision"]
            total["vad_recall"] += outputs["val_recall"]
            total["vad_f1"] += outputs["val_f1"]
            ccounter += 1
        vad_loss = float(total["vad_loss"]) / ccounter
        vad_acc = float(total["vad_acc"]) / ccounter
        vad_precision = float(total["vad_precision"]) / ccounter
        vad_recall = float(total["vad_recall"]) / ccounter
        vad_f1 = float(total["vad_f1"]) / ccounter

    return vad_loss, vad_acc, vad_precision, vad_recall, vad_f1


def trainer(num_epochs, model, loader_dict, optimizer, device, outdir):

    best_val_loss = 1000000000
    for epoch in range(num_epochs):
        print("Epoch:{}".format(epoch+1))
        train_loss, train_acc, train_precision, train_recall, train_f1 = train(model, optimizer, loader_dict['train'], device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = val(model, loader_dict['val'], device)
        print('------------------------------------')
        print("Train loss      : {}".format(train_loss))
        print("Train Acc       : {}".format(train_acc))
        print("Train Precision : {}".format(train_precision))
        print("Train Recall    : {}".format(train_recall))
        print("Train F1        : {}".format(train_f1))
        print('------------------------------------')
        print("Val loss      : {}".format(val_loss))
        print("Val Acc       : {}".format(val_acc))
        print("Val Precision : {}".format(val_precision))
        print("Val Recall    : {}".format(val_recall))
        print("Val F1        : {}".format(val_f1))
        print('------------------------------------')
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, "best_val_loss_model.pth"))
            
def tester(model, loader_dict, device):
    test_loss, test_acc, test_precision, test_recall, test_f1 = val(model, loader_dict['test'], device)
    print('------------------------------------')
    print("Test loss      : {}".format(test_loss))
    print("Test Acc       : {}".format(test_acc))
    print("Test Precision : {}".format(test_precision))
    print("Test Recall    : {}".format(test_recall))
    print("Test F1        : {}".format(test_f1))
    print('------------------------------------')