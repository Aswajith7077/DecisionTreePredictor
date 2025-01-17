from DataHandler import DataHandler


class Cart:

    __prediction_types = ['regression','classification']
    
    def __init__(self,prediction_type:str = 'regression',max_depth:int = 50):

        if prediction_type not in Cart.__prediction_types:
            raise AttributeError('Invalid Prediction Type')
        

        self.prediction_type = prediction_type
        self.max_depth = max_depth


    def fit(self,data:DataHandler):

        pass

