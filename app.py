from DataHandler import DataHandler


d = DataHandler(
    fileName = './data/SRSno-avg.csv',
    extension = 'csv',
    label_column = 'HAS ADHD'
)

print(f'Train Features : \n{d.train_features.head()}')
print(f'Train Labels : \n{d.train_labels.head()}')

print(f'Test Features : \n{d.test_features.head()}')
print(f'Test Labels : \n{d.test_labels.head()}')


print(f'Test info : {d.test_features.info()}')
print(f'Test info : {d.train_features.info()}')