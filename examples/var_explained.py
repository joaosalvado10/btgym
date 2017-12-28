import os
filename='./data/Train_EurUsd.csv'
import pandas as pd
import xgboost

from matplotlib import pyplot

names=['open', 'high', 'low', 'close', 'ir10us','ir2us','ir2eu',
               'ir10eu','spx','oil','yield_10','yield_2']
assert filename and os.path.isfile(filename)
current_dataframe = pd.read_csv(filename,names=names)

print(current_dataframe)


df_train = current_dataframe.iloc[:int(0.75 * len(current_dataframe))]
df_test = current_dataframe.iloc[int(0.75 * len(current_dataframe)):]


y_train = df_train['close']
y_test = df_test['close']
X_train = df_train[['ir10us','ir2us','ir2eu','ir10eu','spx','oil','yield_10','yield_2','open']]
X_test = df_test[['ir10us','ir2us','ir2eu','ir10eu','spx','oil','yield_10','yield_2','open']]


'''
open = 1,
high = 2,
low = 3,
close = 4,
ir10us = 5,
ir2us = 6,
ir2eu = 7,
ir10eu = 8,
spx = 9,
oil = 10,
yield_10 = 11,
yield_2 = 12,
'''

model = xgboost.XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
xgboost.plot_importance(model)
pyplot.show()