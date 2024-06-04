import argparse
import torch
from torch import nn
from model import BERT_Senti
from data_preprocess import data_preprocess,BERTDataset
from torch.utils.data import DataLoader
from train_test import train,test

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Configuration for the training script.')

    # Add arguments
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Stock for training and testing')
    parser.add_argument('--epochs', type=int, default=10, help='Training sequence length.')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to train on.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')

    # Parse arguments
    args = parser.parse_args()

    file_path = f'data/training.1600000.processed.noemoticon.csv.zip'
    df_train,df_test = data_preprocess(file_path)

    def collate_fn(data):
        sequences, labels = zip(*data)
        sequences, labels = list(sequences), list(labels)
        sequences = model.tokenizer(sequences, padding=True, truncation=True,max_length=512, return_tensors="pt")
        return sequences,torch.tensor(labels)

    train_data = BERTDataset(df_train)
    test_data = BERTDataset(df_test)
    train_dataloader = DataLoader(train_data, batch_size= args.batch_size, shuffle=True,collate_fn = collate_fn)
    test_dataloader = DataLoader(test_data, batch_size= args.batch_size, shuffle=True, collate_fn = collate_fn)

    model = BERT_Senti('distilbert-base-uncased')
    model.to(args.device)

    for epoch in range(args.epochs):
        train(
            model=model,
            train_dataloader=train_dataloader, 
            optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            loss_fn=nn.CrossEntropyLoss().to(args.device),
            epoch=epoch,
            args=args
        )
        test(
            model=model,
            test_dataloader=test_dataloader, 
            loss_fn=nn.CrossEntropyLoss().to(args.device),
            epoch=epoch
        )
