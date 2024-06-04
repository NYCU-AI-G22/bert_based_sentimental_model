import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import re


def preprocessing_function(text: str) -> str:

    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    preprocessed_text = re.sub(r'[^\w\s]', '', preprocessed_text)
    preprocessed_text = re.sub(r'<[^>]+>', '', preprocessed_text)

    return preprocessed_text

def data_preprocess(file_path):

    zip_file_path = file_path

    with zipfile.ZipFile(zip_file_path, 'r') as z:
        csv_file_name = z.namelist()[0]  
        with z.open(csv_file_name) as csv_file:
            df = pd.read_csv(csv_file, encoding='utf-8',encoding_errors='ignore')

    df = df.sample(frac=0.01, random_state=42)
    df.rename(columns={'0':'label',df.columns[-1]:'text'},inplace = True)
    df.drop(columns=[column for column in df.columns if column not in ['label','text']],inplace=True)
    df['label'] = df['label'].replace(4, 1)
    df['text'] = df['text'].apply(preprocessing_function)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    return train_df, valid_df

class BERTDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.data = {}
        for idx, row in df.iterrows():
            self.data[idx] = (row['text'], row['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return text, torch.tensor(label)
    
if __name__ == "__main__":
    file_path = f'data/training.1600000.processed.noemoticon.csv.zip'
    data,_ = data_preprocess(file_path)
    dataset = BERTDataset(data)
    print(dataset[0])