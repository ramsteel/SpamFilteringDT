import streamlit as st
import validation.input_validation as validation
import controller.controller as ctr

def check_page(DT, X_train):
    st.write("<h1 style='text-align: center';>Kelompok 6 Keamanan Jaringan<br />Decision Tree Algorithm</h1>",
             unsafe_allow_html=True)
    st.write('---')
    input_msg = st.text_input(label='Input Message',
                              placeholder='Your Email Message')
    btn = st.button('Check', type='primary')
    if btn:
        if validation.length_validation(input_msg):
            ctr.spam_filtering(input_msg, DT, X_train)
        else:
            st.error('*__Tolong Masukkan Pesan!__*')
