from flask import Flask, request, render_template
import torch
import re
from model_class import *  # Ensure this import matches your model's class
from torchtext.data.utils import get_tokenizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


load_path = './model/best_s_bert.pt'
params, state = torch.load(load_path)
model_S_Bert = BERT(**params, device=device).to(device)
model_S_Bert.load_state_dict(state)

# Load the 'basic_english' tokenizer
tokenizer = get_tokenizer('basic_english')
vocab = torch.load('./model/vocab.pt')

def tokenize_and_pad(sentence, tokenizer, vocab, max_seq_length):
    tokens = tokenizer(re.sub("[.,!?\\-]", '', sentence.lower()))
    input_ids = [vocab['[CLS]']] + [vocab[token] for token in tokens] + [vocab['[SEP]']]
    n_pad = max_seq_length - len(input_ids)
    attention_mask = ([1] * len(input_ids)) + ([0] * n_pad)
    input_ids = input_ids + ([0] * n_pad)

    return {'input_ids': torch.LongTensor(input_ids).reshape(1, -1),
            'attention_mask': torch.LongTensor(attention_mask).reshape(1, -1)}

def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool



def calculate_similarity(model, tokenizer, vocab, sentence_a, sentence_b, device):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs_a = tokenize_and_pad(sentence_a, tokenizer, vocab, max_seq_length)
    inputs_b = tokenize_and_pad(sentence_b, tokenizer, vocab, max_seq_length)
    

    # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a['input_ids'].to(device)
    attention_a = inputs_a['attention_mask'].to(device)
    inputs_ids_b = inputs_b['input_ids'].to(device)
    attention_b = inputs_b['attention_mask'].to(device)
    segment_ids = torch.zeros(1, max_seq_length, dtype=torch.int32).to(device)

    # Extract token embeddings from BERT
    u = model.get_last_hidden_state(inputs_ids_a, segment_ids)  
    v = model.get_last_hidden_state(inputs_ids_b, segment_ids) 
    

    # Get the mean-pooled vectors
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score

max_seq_length = 512


@app.route('/', methods=['GET', 'POST'])
def index():
    similarity_score = None
    if request.method == 'POST':
        sentence1 = request.form.get('sentence1')
        sentence2 = request.form.get('sentence2')
        similarity_score = calculate_similarity(model_S_Bert, tokenizer, vocab, sentence1, sentence2, device)
    return render_template('index.html', similarity_score=similarity_score)

if __name__ == '__main__':
    app.run(debug=True)
