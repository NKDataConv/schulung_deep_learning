from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Deep Seek Modell
MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generator = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Create 10 positive movie reviews with the word broccoli: 1. One of the standout moments in this film was the tender, flavorful broccoli served as the heart of a hearty meal. 2."

inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH,
                        truncation=True, padding=True)
inputs = inputs.to("cpu")

outputs = generator.generate(
    **inputs,
    max_length=MAX_LENGTH,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print(outputs)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Extrahiere nur die Antwort nach dem "Antwort:" Teil
answer = generated_text.split("Antwort:")[-1].strip()