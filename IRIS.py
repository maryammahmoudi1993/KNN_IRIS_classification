import pandas as ps
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss

def load_data():
    raw_dataset = ps.read_csv('iris.data', header= None , 
                        names=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','classes'])

    rows , cols = raw_dataset.shape
    print("Number of samples: {} Number of features:{} ".format(rows,cols-1))

    dataset = raw_dataset.iloc[:,:cols-1]
    target = raw_dataset.iloc[:,cols-1]
    """print(dataset)
    print(target)"""
    x_train, x_test, y_train, y_test = train_test_split(dataset,target, test_size=0.1)
    print("Number of train data: {}  number of test data:  {}".format(len(x_train),len(x_test)))
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_data()

def training():
    classifier = KNeighborsClassifier(n_neighbors=1)

    classifier.fit(x_train,y_train)
    
    return classifier

classifier = training()

#print(y_predict == y_test)
#print(y_predict)
def accuracy():
    y_predict = classifier.predict(x_test)
    accuracy = accuracy_score(y_test,y_predict)
    loss = zero_one_loss(y_test,y_predict)
    return accuracy, loss


accuracy,loss = accuracy()
print("Accuracy = %{:.2f}  Loss = {}".format(accuracy*100, loss*100))