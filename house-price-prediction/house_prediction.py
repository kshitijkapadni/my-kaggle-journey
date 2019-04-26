import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

path = "D:\kskML\Kaggle\house prediction\dataset\\train.csv"
data = pd.read_csv(path)

cols = ['MiscFeature','Fence','PoolQC','FireplaceQu','Alley','Utilities','LandContour']
data = data.drop(cols, axis=1)
data_category = data.select_dtypes(['object']).copy()
data = data.drop(data.select_dtypes(['object']),axis=1)
data_category.MasVnrType.fillna(data_category.MasVnrType.value_counts().index[2],inplace=True)
data_category.MasVnrType.replace('None',data_category.MasVnrType.value_counts().index[1],inplace=True)
colss = ['BsmtFinType1','BsmtFinType2','ExterQual','ExterCond','LotShape','BsmtQual','BsmtCond','HeatingQC','KitchenQual','GarageQual','BsmtExposure','GarageCond']     

for col in data_category.columns.values:
    if col in colss:
        data_category[col].fillna('Po',inplace=True)
        data_category[col].replace('None','Po',inplace = True)
        continue
    data_category[col].fillna(data_category[col].value_counts().index[0],inplace=True)
    data_category[col].replace('None',data_category[col].value_counts().index[0],inplace = True)

for col in colss:
    replace_map = {col: {'Reg':4,'IR1':3,'IR2':2,'IR3':1,'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'Av': 3, 'Mn': 2, 'No': 1,'GLQ': 5, 'ALQ': 4,'BLQ':3,'Rec':2,'LwQ':1,'Unf':1}}
    data_category.replace(replace_map,inplace=True)
    
data_category.replace({'LandSlope':{'Gtl':1,'Mod':0.5,'Sev':0},'CentralAir':{'Y':1,'N':0},'PavedDrive':{'Y':1,'P':0.5,'N':0},'Street':{'Pave':1,'Grvl':0}},inplace=True)
data_category = data_category.drop(data_category.select_dtypes(['object']),axis=1)
data = pd.concat([data,data_category],axis=1)

data.replace(np.nan,data.mean(), inplace=True)
y = data.SalePrice
X = data.drop(['Id','SalePrice','MSSubClass','KitchenAbvGr','EnclosedPorch','GarageYrBlt','TotalBsmtSF','CentralAir','PavedDrive','YearRemodAdd','GarageQual','GarageCond','LotShape','ExterCond','LandSlope','BsmtCond','PoolArea','OpenPorchSF','LowQualFinSF','BsmtUnfSF','FullBath','YrSold','MoSold','MiscVal','BsmtHalfBath','BsmtFinSF2','HalfBath','BedroomAbvGr'],axis=1)

scale = StandardScaler()
X[X.columns.values] = scale.fit_transform(X[X.columns.values])
#
est = sm.OLS(y, X).fit()
est.summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 250,random_state=42)

model.fit(X,y)

error = np.sqrt(np.mean(np.square(np.log1p(y_test) - np.log1p(model.predict(X_test)))))
import pickle

out = open('housePred.pickle','wb')
pickle.dump(model,out)
out.close()

