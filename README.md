# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Start the program 
2. Import the python pandas as pd
3. Read the dataset of Salary csv file
4. Display the information of the dataset using info() method
5. Import the LabelEncoder for preprocessing of the dataset
6. Assign x as Position and Level columns values and assign y as Salary column value
7. From sklearn library import the Decision Tree Regressor and predict the x_test
8. Find the mean squared error and print it
9. Print the r2 metrics
10. Predict the Decision tree using random values
11. Stop the program

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: kanisha.B
RegisterNumber: 212219220021
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
data["Level"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](/images/dreg_data_info.png)

![Decision Tree Regressor Model for Predicting the Salary of the Employee](/images/dreg_mse.png)

![Decision Tree Regressor Model for Predicting the Salary of the Employee](/images/dreg_r2.png)

![Decision Tree Regressor Model for Predicting the Salary of the Employee](/images/dreg_predict.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
