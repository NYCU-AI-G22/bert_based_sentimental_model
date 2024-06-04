import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def save_model(model, path):
    torch.save(model.state_dict(), path)

def train(model, train_dataloader, optimizer, loss_fn, epoch, args):

    training_loss = 0
    train_pred = []
    train_label = []  
    total_steps = len(train_dataloader)

    with tqdm(total=total_steps, desc=f"Epoch {epoch+1}", leave=True, ncols=100, unit='step') as pbar:
        for inputs, labels in train_dataloader: 
            model.train()
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            y_pred = model(**inputs)
            loss = loss_fn(y_pred, labels)        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()
            
            train_label.append(labels.detach().to('cpu').numpy())   
            y_pred = y_pred.detach().to('cpu').numpy()    
            y_pred = np.array([arr.argmax() for arr in y_pred])         
            train_pred.append(y_pred)
            pbar.update(1) 
                      
    true = np.concatenate([arr.ravel() for arr in train_label])
    pred = np.concatenate([arr.ravel() for arr in train_pred])
    precision, recall, f1, support = precision_recall_fscore_support(true, pred, average='macro', zero_division=1)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    avg_loss = round(training_loss/len(train_dataloader), 4)
    print(f"Training F1 score: {f1}, Training Precision: {precision}, Training Recall: {recall}, Training Loss: {avg_loss}")
    # Save the trained model
    save_model(model, f"BERT_{args.epochs}trained_model.pt")

def test(model, test_dataloader, loss_fn, epoch):

    testing_loss = 0
    test_pred = []
    test_label = []  
    total_steps = len(test_dataloader)

     # testing stage
    for inputs, labels in test_dataloader: # This is pseudo code
        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            y_pred = model(**inputs)

        loss = loss_fn(y_pred, labels) 
        testing_loss += loss.item()    

        test_label.append(labels.detach().to('cpu').numpy())  
        y_pred = y_pred.detach().to('cpu').numpy()    
        y_pred = np.array([arr.argmax() for arr in y_pred])   
        test_pred.append(y_pred)

    true = np.concatenate([arr.ravel() for arr in test_label])
    pred = np.concatenate([arr.ravel() for arr in test_pred])
    precision, recall, f1, support = precision_recall_fscore_support(true, pred, average='macro', zero_division=1)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    avg_loss = round(testing_loss/len(test_dataloader), 4)
    print(f"Testing F1 score: {f1}, Testing Precision: {precision}, Testing Recall: {recall}, Testing Loss: {avg_loss}")
   
