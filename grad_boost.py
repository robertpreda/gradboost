from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_squared_log_error, r2_score 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/train.csv')
X_Data = df[['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt']]
y = df['SalePrice']

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X_Data, y, test_size = 0.2)

ensemble = []
num_of_learners = int(input("Enter number of learners: "))
depth = int(input("Enter max depth of learners: "))

# training (I think idk I'm stupid pls help)
learner = DecisionTreeRegressor(max_depth = depth)
learner.fit(X_train,y_train)
predictions = 0
predictions += learner.predict(X_test)
ensemble.append(learner)
residue = y_test - predictions
train, test = [],[]
residue_list = [residue]
new_train = residue + predictions
for i in range(1, num_of_learners):
    learner = DecisionTreeRegressor(max_depth = depth)
    
    learner.fit(X_train[:len(residue)],residue)
    predictions += learner.predict(X_test)
    residue = y_test - predictions
    new_train += residue
    residue_list.append(residue)
    ensemble.append(learner)
    train.append(r2_score(y_train, ensemble[i].predict(X_train)))
    test.append(r2_score(y_test, ensemble[i].predict(X_test)))

score = r2_score(y_test,predictions)
print("R2 score: ", score)
'''
plt.plot(train, label = "train",color = 'red')
plt.plot(test,'g',label = "test")
plt.plot(residue,'bo',label = "residue")
plt.grid()
plt.legend()
plt.show()
'''