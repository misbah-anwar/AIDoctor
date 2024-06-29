import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "aboonaji/llama2finetune-v2",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_quant_type="nf4"
        )
    )
    tokenizer = AutoTokenizer.from_pretrained("aboonaji/llama2finetune-v2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Main Streamlit app
def main():
    st.title("AI Doctor")

    model, tokenizer = load_model()
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=300
    )

    user_prompt = st.text_input("Enter your question:")
    
    if user_prompt:
        response = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
        st.write(response[0]['generated_text'])

if __name__ == "__main__":
    main()
