## **Deployed App on Streamlit link :-** [click here](https://yashkumbalkar-mini-english-hindi-translator-app-kezsvf.streamlit.app/)

# English-to-Hindi Translator

### Overview :-

This is an English-to-Hindi translator built using an Encoder-Decoder model with Bahdanau Attention. The model is trained to 
translate English sentences into Hindi. The project is deployed as a web application using Streamlit.

### Data Source :-

The dataset used for this project is sourced from Kaggle:- [English Hindi Dataset](https://www.kaggle.com/datasets/preetviradiya/english-hindi-dataset)

### Project Description :-

The model is trained on a English-Hindi dataset. Preprocessing includes tokenization, padding, and sequence-to-sequence transformation.

### Model Architecture :-

- `Encoder:` Processes the input English sentence and generates a context vector.
- `Bahdanau Attention:` Helps the decoder focus on relevant parts of the input sequence.
- `Decoder:` Generates the translated Hindi sentence word by word.

### Example Usage :-

- Open the deployed Streamlit app.
- Enter an English sentence in the text input field.
- Click the "Translate" button to get the Hindi translation.
