from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV , train_test_split
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

df=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\Ödev\\Iris.csv")
y=df["iris"]
dataFrame=df.drop(["iris"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(dataFrame,y,test_size=0.3,random_state=200)

lgbm=LGBMClassifier()
lgbm_params={
    "learning_rate":[0.01,0.1,0.5],
    "num_leaves":[30,40,50],
    "max_depth":[3,4,5,6],
}

lgbm_cv=GridSearchCV(lgbm,lgbm_params,cv=3,n_jobs=-1,verbose=2)
lgbm_cv.fit(x_train,y_train)
learning_rate=lgbm_cv.best_params_["learning_rate"]
num_leaves=lgbm_cv.best_params_["num_leaves"]
max_depth=lgbm_cv.best_params_["max_depth"]
lgbm_tuned=LGBMClassifier(max_depth=max_depth,learning_rate=learning_rate,num_leaves=num_leaves)
lgbm_tuned.fit(x_train,y_train)
predict=lgbm_tuned.predict(x_test)
for i in predict:
    print(i)
acscore=accuracy_score(y_test,predict)
cm=confusion_matrix(y_test,predict)
print(acscore)
print(cm)


