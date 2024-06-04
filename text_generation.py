import torch
from model import BERT_Senti
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()

def generate_emotional_text(input_text,bert_model,tokenizer):
    inputs = bert_model.tokenizer(input_text, padding=True, truncation=True,max_length=512, return_tensors="pt")
    outputs = bert_model(**inputs)
    sentiment =  torch.argmax(outputs, dim=1).item()

    # 情感修改
    if sentiment == 1: 
        prompt = input_text + " positive sentiment" 
    else:
        prompt = input_text + " negative sentiment"
    
    inputs = tokenizer([prompt], return_tensors='pt')

     # 生成回复
    reply_ids = model.generate(**inputs, max_length=100)
    reply_text = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

    return reply_text

if __name__ == "__main__":
    bert_model = BERT_Senti('distilbert-base-uncased')
    load_model(bert_model, f"BERT_10trained_model.pt") 
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
    input_text = "Today's weather is not so well"
    print(generate_emotional_text(input_text,bert_model,tokenizer))