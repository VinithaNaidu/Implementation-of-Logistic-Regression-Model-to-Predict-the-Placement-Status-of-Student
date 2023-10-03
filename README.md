# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import the standard libraries. 
2.  Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.  LabelEncoder and encode the dataset.
4.  Import LogisticRegression from sklearn and apply the model on the dataset.
5.  Predict the values of array.
6.  Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.  Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S.Shanmathi
RegisterNumber:  212222100049
*/

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#removes the row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1= classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

### Placement data:
![Screenshot 2023-10-02 122542](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/4b409946-0f36-49d8-86d5-02075e561bab)

### Salary data:
![Screenshot 2023-10-02 122558](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/ecf85a05-91e1-443d-baad-d056358ca0d2)

### Checking the null() function:
![Screenshot 2023-10-02 122606](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/f60a7465-2981-48c4-849c-6a90ec5e1c8d)

### Data Duplicate:
![Screenshot 2023-10-02 122614](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/3690e9d1-73be-46d7-b328-4af6e99c210f)

### Print data:
![Screenshot 2023-10-02 122647](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/7e8b67a3-0087-4a81-81aa-65b32f0f2309)

### Data-status:
![Screenshot 2023-10-02 122656](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/d5172fe8-46c1-48b5-a71b-ca425721524b)

### y_prediction array:
![Screenshot 2023-10-02 122715](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/97aa62c3-06ce-4681-a4d1-7fad20bf1486)

### Accuracy value:
![Screenshot 2023-10-02 122722](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/68d9509c-d5b8-44af-b92b-f6f8bb0c2739)

### Confusion array:
![Screenshot 2023-10-02 122728](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/c62f9844-a287-411d-8984-2297b60bb963)

### Classification report:
![Screenshot 2023-10-02 122735](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/4126f809-488d-46f1-8a00-e743a04938b2)

### Prediction of LR:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121243595/10cf704a-f99a-45c6-b3a1-6aec4ea3678d)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
