### Hinweise zur Demo:
# - **BERT (Bidirectional Encoder Representations from Transformers)** wird dafür verwendet, um Kontext und Bedeutung aus dem Eingabetext zu extrahieren. Dabei wird der gesamte Satz berücksichtigt.
# - **GPT-2 (Generative Pre-trained Transformer 2)** ist ein autoregressives Modell, das in der Lage ist, basierend auf einem gegebenen Text neue Texte zu generieren.
# - **Tokenisierung:** Vor der Verwendung müssen Texte in ein geeignetes Format verwandelt werden, was durch Tokenizer geschieht, die Wörter oder Wortteile in numerische IDs umwandeln.


# Importieren der benötigten Bibliotheken
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Funktion zur Durchführung der Inferenz mit BERT
def inference_with_bert(text):
    # Initialisieren des BERT Tokenizers und Modells
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenisierung der Eingabetexte: Umwandlung in IDs und Erstellen von Attention Masken
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Inferenz durchführen
    outputs = model(**inputs)

    # Ausgabe der BERT-Embeddings
    return outputs.last_hidden_state


# Funktion zur Erstellung von Text mit GPT-2
def generate_text_with_gpt(text, max_length=50):
    # Initialisieren des GPT-2 Tokenizers und Modells
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Tokenisierung des Eingabetextes
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Textgenerierung
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Dekodierung der Ausgabe-IDs in lesbaren Text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def generate_text(text, max_length=50):
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "lightblue/lb-reranker-0.5B-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenisierung des Eingabetextes
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Textgenerierung
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Dekodierung der Ausgabe-IDs in lesbaren Text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


# Beispieltext für BERT
bert_text = "Deep Learning has revolutionized the field of artificial intelligence."
print("Performing inference with BERT...")
bert_output = inference_with_bert(bert_text)
print("BERT Output Shape:", bert_output.shape)  # Ausgabe der Form der BERT-Embeddings

# Beispieltext für GPT-2
gpt_text = "Once upon a time in a distant land"
print("\nGenerating text with GPT-2...")
gpt_generated_text = generate_text_with_gpt(gpt_text)
print("Generated Text with GPT-2:", gpt_generated_text)


gpt_text = "Once upon a time in a distant land"
print("\nGenerating text with GPT-2...")
gpt_generated_text = generate_text(gpt_text)
print("Generated Text with GPT-2:", gpt_generated_text)