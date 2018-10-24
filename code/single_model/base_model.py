import numpy as np
import pandas as pd

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn


def data_preprocess(train,test):
    
    # Get outlier indexes (2 features: GrLivArea and GarageArea)
    outlier_idx = list(train[train['GrLivArea'] >= 4000].index) + list(train[train['GarageArea'] >= 1200].index)

    # Remove outliers:
    train.drop(train.index[outlier_idx],inplace=True)
    
    # Concat: train + test
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))
    

    # Features (contain a lots of MISSING VALUES) need to be removed.
    to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
    
    # Remove 
    all_data = all_data.drop(to_delete,axis=1)


    # LogTransform for Skewed_Feature (SalePrice)
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # Handle categorical features:
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())        


    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    return X_train,X_test,y
    

# read data
train = pd.read_csv("../../input/train.csv")
test = pd.read_csv("../../input/test.csv") 

Xtrain,Xtest,ytrain = data_preprocess(train,test)

