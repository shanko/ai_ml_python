from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load NLLB model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a translation pipeline
translator = pipeline("translation", model=model, tokenizer=tokenizer)

# Example input
text = "Good morning, how are you?"
text = "सुप्रभात, तुम्ही कसे आहात?"
# Translate English → Marathi
translated = translator(text, tgt_lang="eng_Latn", src_lang="mar_Deva")

print("Input:", text)
print("Output:", translated[0]['translation_text'])
