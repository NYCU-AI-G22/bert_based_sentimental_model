import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BERT_Senti(nn.Module):

    def __init__(self, pretrained_type):
        super().__init__()

        num_labels = 2
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_type)
        self.pretrained_model = AutoModel.from_pretrained(pretrained_type, num_labels=num_labels)

        self.classifier = nn.Sequential(
            nn.Linear(self.pretrained_model.config.hidden_size, 512),  
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(128, 2)
        )

    def forward(self, **pretrained_text):
        outputs = self.pretrained_model(**pretrained_text).last_hidden_state
        pretrained_output = outputs[:, 0, :]
        logits = self.classifier(pretrained_output)

        return logits


