# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:30:16 2018

@author: Administrator
"""
from sklearn.model_selection import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
#导入数据预处理，包括标准化处理或正则处理
from sklearn import preprocessing
#样本平均测试，评分更加
from sklearn.cross_validation import cross_val_score
 
from sklearn import datasets
#导入knn分类器
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#数据预处理
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
#用于训练数据和测试数据分类
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib.font_manager import FontProperties

font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)

trees=100
#excel文件名
fileName="data1.xlsx"
#fileName="GermanData_total.xlsx"
#读取excel
df=pd.read_excel(fileName)
# data为Excel前几列数据
x1=df[df.columns[:-1]]
#标签为Excel最后一列数据
y1=df[df.columns[-1:]]
 
#把dataframe 格式转换为阵列
x1=np.array(x1)
y1=np.array(y1)
#数据预处理，否则计算出错
y1=[i[0] for i in y1]
y1=np.array(y1)

#数据预处理
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) 
imp.fit(x1)
x1=imp.transform(x1)

forest=RandomForestClassifier(n_estimators=trees,random_state=0)

x_train,x_test,y_train,y_test=train_test_split(x1,y1,random_state=0)
forest.fit(x_train,y_train)
print("accuracy on the training subset:{:.3f}".format(forest.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(forest.score(x_test,y_test)))
print('Feature importances:{}'.format(forest.feature_importances_))


feature_names=list(df.columns[:-1])
n_features=x1.shape[1]
plt.barh(range(n_features),forest.feature_importances_,align='center')
plt.yticks(np.arange(n_features),feature_names,fontproperties=font)
plt.title("random forest with %d trees:"%trees)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()




