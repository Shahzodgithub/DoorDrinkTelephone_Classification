import streamlit as st
from fastai.vision.all import *

# title 
st.title("Eshik,Telefon,Ichimlik larni aniqlovchi model")
file = st.file_uploader('Ramni yuklang', type=['png','jpeg','gif','svg'])
if file: 
    st.image(file)
    img = PILImage.create(file)

    #model yuklash 
    model = load_learner('DoorDrinkTelephone.pkl')

    # model baholash
    pred,pred_id, probs = model.predict(img)
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100: .1f}%')