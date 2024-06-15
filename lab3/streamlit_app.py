import streamlit as st
import torch
from transformers import pipeline
from transformers import VitsModel, AutoTokenizer


@st.cache_resource()
def load_model():
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    return tokenizer, model

st.title("Streamlit текст-в-речь переводчик!")
text = st.text_input("Введите текст:", value="some example text in the English language")
button_submit = st.button("Распознать текст")
tokenizer, model = load_model()
    
if button_submit:
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform
        
        st.audio(output.float().numpy(), sample_rate=model.config.sampling_rate)
        st.snow()
