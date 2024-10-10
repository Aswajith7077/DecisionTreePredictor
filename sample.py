import streamlit as st
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")


file_formats = ['csv','xls','xlsx','xlsb','ods','xlsm']

if uploaded_file is not None:
    
    
    format = uploaded_file.name.split('.')[-1]
    print(format)
    if format == 'csv':

        st.text_input('Delimiter')


        
    elif format in file_formats:
        
        dataframe = pd.read_csv(uploaded_file)
        print(dataframe)
        st.write(dataframe)
    
    


        