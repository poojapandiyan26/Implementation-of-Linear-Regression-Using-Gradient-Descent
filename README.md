# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Intialize weights randomly.
2.Compute predicted.
3. gradient of loss function.
4.Update weights using gradient descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: P.POOJA SRI
RegisterNumber:  212224230197
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("//content//50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![Screenshot 2025-03-17 150447](https://github.com/user-attachments/assets/962873cd-0062-45d6-b4bf-f3a3777fa31f)
![Screenshot 2025-03-17 150457](https://github.com/user-attachments/assets/d4490795-8f6a-400b-bc88-89ebef80f6fb)
![Screenshot 2025-03-17 150504](https://github.com/user-attachments/assets/31a1358f-0cba-4977-94e3-ae5e119fa7d5)
![Screenshot 2025-03-17 150512](https://github.com/user-attachments/assets/d53065a3-e895-4520-906d-97d9063bdf8d)
![Screenshot 2025-03-17 150525](https://github.com/user-attachments/assets/9036210f-57fc-4f8c-8d38-8526f4687d5e)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
