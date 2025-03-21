import streamlit as st
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F

from preprocess import clean_text

# Load the vocabularies
features_vocab = torch.load('features_vocab.pth')
target_vocab = torch.load('target_vocab.pth')

# Set device to cpu
device = 'cpu'

# Defining Model Architecture

# Encoder
class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

# Attention
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()

        self.Wa = nn.Linear(hidden_size, hidden_size)   # Transform for query
        self.Ua = nn.Linear(hidden_size, hidden_size)   # Transform for keys
        self.Va = nn.Linear(hidden_size, 1)   # Compute the attention score

    def forward(self, query, keys):
        key_shape = keys.size()

        query = query.repeat(1, key_shape[1], 1)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(-1)

        weights = torch.softmax(scores, dim=1)
        weights = weights.unsqueeze(1)
        context = torch.bmm(weights, keys)

        return context, weights

# Decoder
class AttnDecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderGRU, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)

        bos_token_index = features_vocab['<bos>']
        decoder_input = torch.full((batch_size, 1), bos_token_index, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        # fixed length set to 5
        for i in range(5):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

# Load the Models
encoder = torch.load('encoder_epoch_500_new.pth', map_location=torch.device('cpu'))
decoder = torch.load('decoder_epoch_500_new.pth', map_location=torch.device('cpu'))


# Prediction

tokenizer_eng = get_tokenizer('basic_english')

def tokens_to_indices_test(tokenized_texts, vocab):
    indices_texts = []
    for sentence in tokenized_texts:
        indices_texts.append([vocab[token]  if token in vocab else vocab['<unk>'] for token in sentence])
    return indices_texts

def evaluate(encoder, decoder, input_text, feature_vocab, target_vocab):
    with torch.no_grad():
        preprocess = clean_text(input_text)
        tokenized_english_txt_test = tokenizer_eng(preprocess)

        english_indices_test = tokens_to_indices_test([tokenized_english_txt_test], features_vocab)
        input_tensor = torch.LongTensor(english_indices_test[0]).to(device).reshape(1,len(sentence.split()))

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        EOS_token = feature_vocab['<eos>']
        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<eos>')
                break
            decoded_words.append(target_vocab.get_itos()[idx] if idx < len(target_vocab) else '<unk>')
    return decoded_words, decoder_attn



# Streamlit UI
st.set_page_config(page_title="English to Hindi Translator", layout="centered")

st.title("English to Hindi Translator")
st.write("Enter a sequence of words in English to translate it into Hindi")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>English to Hindi Translator</h1>",
    unsafe_allow_html=True,
)


st.markdown(
    """
    ### Welcome to the Translator
    This application translates the english sentence in a hindi sentence based on the text you provide. Powered by a trained Encoder-Decoder model.

    ### How to Use:
    1. Type a sentence or a few words.
    2. Click "Translate".
    """
)




# User input text box
input_text = st.text_area("Enter text", height=100, placeholder="Type here...")


# Predict button
if st.button("Translate"):
    if input_text.strip():
        with st.spinner("Translating..."):
            decoder_output, attn_weights = evluate(encoder, decoder, input_text, features_vocab, target_vocab)
            translate = ''
            for i in decoder_output:
                if i == '<pad>':
                    break
                translate = translate + i + ' '
    else:
        st.warning("Please enter text to translate.")

# Display output in a text area
st.markdown("### Hindi Translation:")
st.text_area(" ", value=translated_text, height=100, disabled=True)

































