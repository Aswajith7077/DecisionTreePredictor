import streamlit as st
import pandas as pd
from io import BytesIO
from DataHandler import DataHandler
from CART_Classifier import CartClassifier



confusion_matrix_path = './data/confusion_matrix.png'

prediction_type = st.sidebar.selectbox(
    "What kind action you are going to perform?",
    ("Regression", "Classification")
)
st.title("Decision Tree Regressor" if(prediction_type == 'Regression') else "Decision Tree Classifier")
uploaded_file = st.file_uploader("Choose a file")




# print(uploaded_file)

file_formats = ['csv','xls','xlsx','xlsb','ods','xlsm']

if uploaded_file is not None and DataHandler.check(uploaded_file.type):

    
    format = uploaded_file.name.split('.')[-1]
    
    # print(format)
    if format == 'csv':
        
        data = DataHandler(uploaded_file,'csv')
        columns = data.getColumns()


        st.header("The Data Frame")
        st.dataframe(data.getDataFrame(),use_container_width = True)
        
        numerical_selected = None
        categorical_selected = None
        label_selected = None
        

        numerical_selected = st.multiselect(
            "Select the Numerical Data",
            columns,
            []
        )

        categorical_selected = st.multiselect(
            "Select the Categorical Data",
            list(set(columns) - set(numerical_selected)),
            []
        )

        label_selected = st.selectbox(
            "Select the label for the data set",
            columns,
        )



        if numerical_selected is not None and categorical_selected is not None and label_selected is not None:
            data.split_data(label_column = label_selected)

            data.setFeatureMap(
                numerical_data = numerical_selected,
                categorical_data = categorical_selected
            )

            col1,col2,col3 = st.columns([50,50,50])

            criterion_selected = col1.selectbox(
                "Select the Criterion",
                ["Gini","Entropy"]
            )
            max_depth_selected = col2.number_input(
                "Select the Max Depth",
                value = 50,
                step = 1,
            )      
            min_samples_selected = col3.number_input(
                "Select the minimum number of samples",
                value = 1,
                step = 1,
            )      

            col1,col2 = st.columns([80,20])

            col1.header("Training Data")
            col1.markdown('Training Features')
            col1.dataframe(data.getTrainFeatures(),use_container_width = True)
            col2.header(" ")
            col2.markdown('Training Labels')
            col2.dataframe(data.getTrainLabels(),use_container_width = True)

            col1,col2 = st.columns([80,20])
            
            col1.header('Testing Data')
            col1.markdown('Testing Features')
            col1.dataframe(data.getTestFeatures(),use_container_width = True)
            col2.header(" ")
            col2.markdown('Testing Labels')
            col2.dataframe(data.getTestLabels(),use_container_width = True)

            if st.button('Submit',type = 'primary'):
                c = CartClassifier(max_depth = max_depth_selected,min_samples = min_samples_selected,criterion = criterion_selected.lower())
                c.fit(data = data)

                fileName:str = './data/temp.py'

                with open(fileName,'w') as file:
                    file.write('import sys\n\ndef predict(x):\n')

                # print('Myroot : ',c.root)

                c.traverse_tree(data = data,node = c.root,fileName = fileName)


                with open(fileName,'a') as file:
                    file.write('\nx = float(sys.argv[1])\nresult = predict(x)')

                with open(fileName,'r') as file:
                    result = file.read()

                st.header('The Code Generation :')
                st.code(result,language='python',line_numbers=True,wrap_lines=False)

                p,r,f1 = c.evaluate(data = data,conf_path = confusion_matrix_path)

                st.title('Report')

                

                col1,col2 = st.columns([5,5])
                with col1:

                    st.write('The Confusion Matrix :')
                    st.image(confusion_matrix_path,'The Confusion Matrix')

                with col2:
                    c1,c2 = st.columns([5,5])
                    c1.header('Precision : ')
                    c1.header('Recall : ')
                    c1.header('F1 Score : ')

                    c2.header(p)
                    c2.header(r)
                    c2.header(f1)

                
        


        
    # elif format in file_formats:
        
    #     dataframe = pd.read_csv(uploaded_file)
    #     print(dataframe)
    #     st.write(dataframe)
    
    


        