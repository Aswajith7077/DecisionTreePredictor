from CART import Cart
from DataHandler import DataHandler
import numpy
from Node import Node
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,ConfusionMatrixDisplay



class CartClassifier:
    
    __possiblecriterion = ['gini','entropy']
    def __init__(self,max_depth:int,min_samples:int = 1,criterion = 'gini'):

        if criterion not in self.__possiblecriterion:
            raise AttributeError('Invalid Criterion Passed')

        if min_samples < 0:
            raise AttributeError('Invalid Min Samples')
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples = min_samples

        
    def getCriterionValue(self,prob:list):

        if self.criterion == CartClassifier.__possiblecriterion[0]:
            return (1 - numpy.sum(numpy.square(prob)))
        else:
            return (-numpy.sum(prob * numpy.log(prob)))

        

    def fit(self,data:DataHandler):


        d = data.data
        self.root = Node(data.train_data)


        children = 1
        levels = 0
        

        terminals = [self.root]
        while levels <= self.max_depth and terminals:

            current_node = terminals[0]
            terminals.pop(0)


            print(f'\n\n\tDepth : {levels}\n')
            

            # print(current_node.data_points[data.label_name].unique())
            # print((len(current_node.data_points[data.label_name].unique()) == 1),(current_node.data_points.shape[0] < self.min_samples))
            if len(current_node.data_points[data.label_name].unique()) == 1:
                # Terminal Node
                
                current_node.associated_class = current_node.data_points[data.label_name].unique()[0]
                print('%8s Ternimal Node Identified! Class : %s\n'%("",str(current_node.associated_class)))
                children -= 1
                continue

            if current_node.data_points.shape[0] < self.min_samples:
                # Terminal Node
                majority_class = current_node.data_points[data.label_name].mode()
                if majority_class.empty:
                    current_node.associated_class = data.classes[0]
                else:
                    current_node.associated_class = majority_class[0]
                print('%8s Ternimal Node Identified! Class : %s'%("",str(majority_class)))
                children -= 1
                continue
            

            thresholds = []
            weighted_ginis = []



            

            for feature in current_node.data_points.columns[:-1]:
                temp_data = current_node.data_points.sort_values(by = feature)
                for i in range(1,temp_data.shape[0]):
                    threshold = (temp_data.iloc[i - 1][feature] + temp_data.iloc[i][feature])/2
                    thresholds.append(threshold)

                    lt = temp_data.loc[temp_data[feature] <= threshold]
                    gt = temp_data.loc[temp_data[feature] > threshold]

                    # print(data.classes)

                    lt_counts = []
                    gt_counts = []

                    for c in data.classes:
                        if c in lt[data.label_name].value_counts().keys():
                            lt_counts.append(lt[data.label_name].value_counts()[c])
                        else:
                            lt_counts.append(0)
                        if c in gt[data.label_name].value_counts().keys():
                            gt_counts.append(gt[data.label_name].value_counts()[c])
                        else:
                            gt_counts.append(0)

                    # print()
                    # print(lt_counts,gt_counts)

                    lt_prob = (numpy.array(lt_counts,dtype = numpy.float32) / lt.shape[0]) if (lt.shape[0] > 0) else numpy.zeros(len(data.classes))
                    gt_prob = (numpy.array(gt_counts,dtype = numpy.float32) / gt.shape[0]) if (gt.shape[0] > 0) else numpy.zeros(len(data.classes))

                    # print(lt_prob,gt_prob)

                    gini_lt = self.getCriterionValue(lt_prob)
                    gini_gt = self.getCriterionValue(gt_prob)

                    # print(gini_lt,gini_gt)

                    weighted_gini = (gini_lt * lt.shape[0] + gini_gt * gt.shape[0])/(lt.shape[0] + gt.shape[0])

                    # print(weighted_gini)
                    weighted_ginis.append(weighted_gini)


                    

                # print(temp_data)
            
                print('\n\n')
                print(threshold,len(weighted_ginis),len(thresholds),lt.shape,gt.shape)
                

            index = numpy.argmin(weighted_ginis)
            chosen_threshold = thresholds[index]
            chosen_weighted_gini = weighted_ginis[index]

            print(f'Chosen Threshold : {chosen_threshold} Weighted Gini : {chosen_weighted_gini}')


            chosen_feature = current_node.data_points.columns[int(index / (current_node.data_points.shape[0] - 1))]

            node_left = Node(
                data_points = current_node.data_points[current_node.data_points[chosen_feature] <= chosen_threshold],
            )

            node_right = Node(
                data_points = current_node.data_points[current_node.data_points[chosen_feature] > chosen_threshold],
            )

            current_node.left = node_left
            current_node.right = node_right
            current_node.threshold = chosen_threshold
            current_node.weighted_gini = chosen_weighted_gini
            current_node.feature = chosen_feature

            terminals.append(node_left)
            terminals.append(node_right)

            children -= 1


            if children == 0:
                children = len(terminals)
                levels += 1

            print(terminals)
        
        for current_node in terminals:

            if len(current_node.data_points[data.label_name].unique()) == 1:
                # Terminal Node

                current_node.associated_class = current_node.data_points[data.label_name].unique()[0]
                children -= 1
                continue

            else:
                # Terminal Node
                majority_class = current_node.data_points[data.label_name].mode()
                if majority_class.empty:
                    current_node.associated_class = data.classes[0]
                else:
                    current_node.associated_class = majority_class[0]
                children -= 1


        

    def traverse_tree(self,data:DataHandler,node:Node,fileName:str, level:int=1):
        if node is None:
            return
        
        indent = " " * (level * 4)
        if node.associated_class != -1:

            print(f"{indent}Level {level}: Leaf Node -> Class = {node.associated_class}")
            with open(fileName,'a') as file:
                file.write(f"\n{indent}return {node.associated_class}")
            
        else:

            print(f"{indent}Level {level}: Threshold = {node.threshold}, Feature = {node.feature}, Gini/Entropy = {node.weighted_gini}")
            with open(fileName,'a') as file:
                file.write(f"\n{indent}if x[{numpy.where(data.data.columns == node.feature)[0][0]}] <= {node.threshold}:")
        
        self.traverse_tree(data = data,node = node.left,fileName = fileName,level = level + 1)

        if node.associated_class == -1:
            with open(fileName,'a') as file:
                file.write(f"\n{indent}else:")
        self.traverse_tree(data = data,node = node.right,fileName = fileName, level = level + 1)


    
    def travel(self,data:DataHandler,node:Node,x):

        if node == None:
            return -1

        if node.associated_class != -1:
            return node.associated_class
        
        feature_index = numpy.where(data.data.columns == node.feature)[0][0]
        if x[feature_index] <= node.threshold:
            return self.travel(data,node.left,x)
        else:
            return self.travel(data,node.right,x)
        

        



    def evaluate(self,data:DataHandler,conf_path:str):

        y_actual = data.getTestLabels()
        features = data.getTestFeatures()


        y_predict = []


        for i in range(features.shape[0]):

            
            x = features.iloc[i].values
            y_predict.append(self.travel(data = data,node = self.root,x = x))
        
        cm = confusion_matrix(y_pred = y_predict,y_true = y_actual)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig(conf_path)

        r = recall_score(y_pred = y_predict,y_true = y_actual)
        p = precision_score(y_pred = y_predict,y_true = y_actual)
        f1 = f1_score(y_pred = y_predict,y_true = y_actual)

        


        return p,r,f1





# Example usage, assuming `classifier.root` is the root of the trained tree