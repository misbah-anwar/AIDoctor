import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("aboonaji/llama2finetune-v2")
    tokenizer = AutoTokenizer.from_pretrained("aboonaji/llama2finetune-v2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

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
