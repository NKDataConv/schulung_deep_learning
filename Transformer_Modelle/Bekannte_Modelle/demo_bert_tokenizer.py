from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample text to tokenize
text = "Hello, how are you?"

# Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)  # Output: ['hello', ',', 'how', 'are', 'you', '?']

# Convert tokens to input IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Input IDs:", input_ids)  # Output: [7592, 1010, 2129, 2024, 2017, 1029]

# Add special tokens (e.g., [CLS] and [SEP]) and convert to input IDs
input_ids_with_special_tokens = tokenizer.encode(text, add_special_tokens=True)
print("Input IDs with special tokens:", input_ids_with_special_tokens)
# Output: [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]

# Decode the input IDs back to text
decoded_text = tokenizer.decode(input_ids_with_special_tokens)
print("Decoded text:", decoded_text)  # Output: "[CLS] hello, how are you? [SEP]"

# Important options:
# - add_special_tokens: Whether to add special tokens like [CLS] and [SEP].
# - max_length: Maximum length of the tokenized input.
# - padding: Whether to pad the input to a certain length.
# - truncation: Whether to truncate the input to a certain length.

# Example with padding and truncation
encoded_input = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=10,
    padding='max_length',
    truncation=True
)
print("Encoded input with padding and truncation:", encoded_input)
# Output: {'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]}

inputs = tokenizer(text, max_length=10, padding="max_length", truncation=True)
print(inputs)
# Output: {'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]}
