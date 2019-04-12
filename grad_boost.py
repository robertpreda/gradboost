from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_squared_log_error, r2_score 
import numpy as np
import pandas as pd

df = pd.read_csv('./data/train.csv')
X_Data = df[['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt']]
y = df['SalePrice']

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X_Data, y, test_size = 0.2)

ensemble = []
num_of_learners = int(input("Enter number of learners: "))
depth = int(input("Enter max depth of learners: "))
# creating the ensemble of learners
for i in range(num_of_learners):
    ensemble.append(DecisionTreeRegressor(max_depth = 3))

# training (I think idk I'm stupid pls help)
ensemble[0].fit(X_train,y_train)
predictions = ensemble[0].predict(X_test)
print("Predictions: ", len(predictions))
print("Shape of predictions: ",predictions.shape)
print("Y shape: ", y_test.shape)
residue = y_test - predictions
for i in range(1, num_of_learners):
    ensemble[i].fit(X_train[:len(residue)], residue)
    predictions = ensemble[i].predict(X_test)
    residue = y_test - predictions
    valid = r2_score(y_test, ensemble[i].predict(X_test))
    print("Validation: ", valid)
