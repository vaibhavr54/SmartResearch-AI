from transformers import T5Tokenizer, T5ForConditionalGeneration

# ✅ Use better model (important)
MODEL_NAME = "google/flan-t5-base"

# Load model once
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def t5_summarize(text):
    try:
        # Clean input
        text = text.replace("\n", " ").strip()

        # Add task prefix
        input_text = "summarize: " + text

        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        outputs = model.generate(
            inputs,
            max_length=120,
            min_length=40,
            num_beams=4,
            no_repeat_ngram_size=3,   # 🚀 prevents repetition
            length_penalty=1.5,
            early_stopping=True
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summary

    except Exception as e:
        return f"Error in T5: {str(e)}"