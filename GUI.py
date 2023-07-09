import streamlit as st
from main import *

# eng_sentence = st.text_input(label='Your English sentence:', placeholder='Type here')
# st.write('The current movie title is', eng_sentence)
SRC, TARGET = build()
# run()

model = torch.load('translate_en_vi.pt')
eng_sentence = st.text_input(label='Your English sentence:', placeholder='Type here')
vi_sentence = translate(model,eng_sentence ,SRC,TARGET,myTokenizerEN)
st.write('Vietnamese sentence: ', vi_sentence)