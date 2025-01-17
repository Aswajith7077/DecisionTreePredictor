import os
import pandas

class DataHandler:
    
    __file_formats = ['csv','xls','xlsx','xlsb','ods','xlsm']
    
    def __init__(self,file,extension:str,test_size:float = 0.2):
        
        # if not os.path.exists(file):
        #     raise FileExistsError(f'The Given file name : {file} does not exists.')
        if test_size <= 0 or test_size >= 1:
            raise ValueError(f'Invalid Test Size : {test_size}')
        
        if extension == DataHandler.__file_formats[0]:
            self.data = pandas.read_csv(file)
        elif extension in DataHandler.__file_formats:
            self.data = pandas.read_excel(file)

        self.test_size = test_size
            
        self.data = self.data.sample(frac = 1)
        
        
        self.data.info()
        self.data.describe()
        self.data.head()
    
    def getDataFrame(self):
        return self.data
    
    def getColumns(self):
        return self.data.columns
    
        

    def check(file_type:str):
        for types in DataHandler.__file_formats:
            if file_type.endswith(types):
                return True
        return False
    

    def split_data(self,label_column:str):

        self.label_name = label_column

        self.classes = self.data[label_column].value_counts().keys()

        self.test_size = int(self.test_size * self.data.shape[0])
        
        self.train_labels = self.data.iloc[:self.data.shape[0] - self.test_size][label_column]
        self.test_labels = self.data.iloc[:self.test_size][label_column]
        
        data = self.data.drop(columns = [label_column])

        self.train_features = self.data.iloc[:data.shape[0] - self.test_size,:]
        self.test_features = self.data.iloc[:self.test_size,:]
        
        self.train_data = self.data.iloc[:self.data.shape[0] - self.test_size,:]
        self.test_data = self.data.iloc[:self.test_size,:]

    def setFeatureMap(self,numerical_data:list,categorical_data:list):
        self.feature_map = {}
        for key in numerical_data:
            self.feature_map[key] = True
        
        for key in categorical_data:
            self.feature_map[key] = False


    def getTrainFeatures(self):
        return self.train_features.head()

    def getTrainLabels(self):
        return self.train_labels.head()
    
    def getTestFeatures(self):
        return self.test_features.head()
    
    def getTestLabels(self):
        return self.test_labels.head()

    def sort_features(self,column_name:str):
        return self.data.sort_values(by = column_name)
    


        

    

