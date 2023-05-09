import model.decision_tree as dt
import streamlit as st


def spam_filtering(msg, dt_train, X_train):
    msg = dt.clean_msg(msg)
    X_train_vct, X_test_vct = dt.tfidf_feature(X_train, [msg])
    pred = dt_train.predict(X_test_vct)
    if pred[0] == 0:
        st.write(':green[NON-SPAM(HAM)]')
    else:
        st.write(':red[SPAM!!]')
