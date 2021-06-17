# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# load the dataset
df = pd.read_csv('data/stockdata.csv', encoding='ISO-8859-1')

# Data cleaning and Preprocessing

### Divide the dataset into train and test
train = df[df['Date'] < '20141231']
test = df[df['Date'] > '20150101']

### Removing punctuations in train dataset
data = train.iloc[:, 2:27]
data.replace('[^a-zA-Z]', ' ', regex = True, inplace = True)

### Renaming columns name from string to numbers
lst = [i for i in range(25)]
new_index = [str(i) for i in lst]
data.columns = new_index

### Converting headlines to lowercase
for index in new_index:
    data[index] = data[index].str.lower()
    
### Now convert all these headlines of a row into one corpus
headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))
    
### Implement Bag Of Words
countvector = CountVectorizer(ngram_range = (2,2))
train_dataset = countvector.fit_transform(headlines)

### Training model using random forest classifier
randomclassifier = RandomForestClassifier(n_estimators=200, criterion = 'entropy')
randomclassifier.fit(train_dataset, train['Label'])


### Predict for Test dataset
test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

### confusion matrix to compare test data with predicted data
cm = confusion_matrix(test['Label'], predictions)
print(cm)

### accuracy score to find accuracy
accuracy = accuracy_score(test['Label'], predictions)
print(accuracy)

### Import classification report
report = classification_report(test['Label'], predictions)
print(report)






