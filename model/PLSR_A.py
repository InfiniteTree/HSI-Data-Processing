from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

data = pd.read_csv("LearningData/TrainData.csv")
print("Dataset loaded...")
x = data.drop(['SPAD',"A1200", "N", "Ca", "Cb"],axis=1)
y1 = data[[ 'Ca', 'Cb']].copy()
x = pd.DataFrame(x, dtype='float32')
y1 = pd.DataFrame(y1, dtype='float32')


#train_x, test_x, train_y, test_y = train_test_split(x,y1,test_size=0.2)
train_x = x[:165]
train_y = y1[:165]

# pls_param_grid = {'n_components': list(range(10,20))}
pls_param_grid = {'n_components':[10]}  
pls = GridSearchCV(PLSRegression(), param_grid=pls_param_grid,scoring='r2',cv=10)

#y_pre = pls.predict(test_x)

test_x = x.tail(81) # u
#print(test_x)
pls.fit(train_x, train_y)
y_pre = pls.predict(test_x)
new_data = y_pre

data.iloc[-81:,-2:] = new_data

data.to_csv('LearningData/NewTrainData.csv', mode='a', header=True, index=False)

'''
train_r2=r2_score(train_y,y_pre)
train_mse = mean_squared_error(train_y, y_pre)
train_RMSE=np.sqrt(train_mse)
y_test_pre = pls.predict(test_x)
test_r2=r2_score(test_y, y_test_pre)
test_mse = mean_squared_error(test_y, y_test_pre)
test_RMSE=np.sqrt(test_mse)
mae = mean_absolute_error(test_y, y_test_pre)
rpd = np.std(test_y) / test_RMSE
print("训练R2:",train_r2)
print("训练RMSE:",train_RMSE)
print("测试R2:",test_r2)
print("测试RMSE:",test_RMSE)
print("测试MSE:",test_mse)
print("测试MAE = ",mae)
print("RPD =", rpd)
'''
# F

