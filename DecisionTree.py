import DataHandler
import pandas
from abc import abstractclassmethod

class DecisionTree:
    
    __algorithms = ['cart','id3','c405']
    
    # def __init__(self,algorithm_type:str = 'cart'):
        
    #     if algorithm_type not in DecisionTree.__algorithms:
    #         raise AttributeError(f'The Algorithm Type : {algorithm_type} not found.')
        
    #     self.__algorithm_type = algorithm_type

    @abstractclassmethod
    def train(self,data:DataHandler) -> None:
        pass
    
    
    def predict(self,data:DataHandler) -> list:
        pass
        
        
        
        
        