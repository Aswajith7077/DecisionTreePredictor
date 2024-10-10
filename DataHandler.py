import os
import pandas

class DataHandler:
    
    __file_formats = ['csv','xls','xlsx','xlsb','ods','xlsm']
    
    def __init__(self,fileName:str,extension:str,delimiter:str = ',',label_column:str = '',test_size:float = 0.2):
        
        if not os.path.exists(fileName):
            raise FileExistsError(f'The Given file name : {fileName} does not exists.')
        elif test_size <= 0 or test_size >= 1:
            raise ValueError(f'Invalid Test Size : {test_size}')
        
        if extension == 'csv':
            data = pandas.read_csv(fileName,delimiter = delimiter)
        elif extension in DataHandler.__file_formats:
            data = pandas.read_excel(fileName)
            
        data = data.sample(frac = 1)
        
        
        data.info()
        
        test_size = int(test_size * data.shape[0])
        
        self.train_labels = data.iloc[:data.shape[0] - test_size][label_column]
        self.test_labels = data.iloc[:test_size][label_column]
        
        data = data.drop(columns = [label_column])
        
        self.train_features = data.iloc[:data.shape[0] - test_size,:]
        self.test_features = data.iloc[:test_size,:]


        

    

