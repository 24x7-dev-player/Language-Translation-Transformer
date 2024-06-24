from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, AutoModelForSeq2SeqLM, AutoConfig
import torch

# Load tokenizers from JSON files
tokenizer_en = Tokenizer.from_file('Tokenizer_en.json')
tokenizer_hi = Tokenizer.from_file('Tokenizer_hi.json')

# Wrap them in a PreTrainedTokenizerFast
tokenizer_en = PreTrainedTokenizerFast(tokenizer_object=tokenizer_en)
tokenizer_hi = PreTrainedTokenizerFast(tokenizer_object=tokenizer_hi)

# Load the model configuration
config = AutoConfig.from_pretrained('model_config')  # Path to the directory containing config.json

# Initialize the model with the config
model = AutoModelForSeq2SeqLM(config)

# Load the model weights
state_dict = torch.load('Weights.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Function to translate English text to Hindi
def translate(text):
    # Tokenize the English text
    inputs = tokenizer_en(text, return_tensors="pt")

    # Generate translation using the model
    translated_tokens = model.generate(**inputs)

    # Decode the tokens to Hindi text
    translated_text = tokenizer_hi.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_text

# Example usage
english_text = "Hello, how are you?"
hindi_translation = translate(english_text)
print(hindi_translation)
