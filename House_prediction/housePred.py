import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import log
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def featureAnalysis(df_train, df_test):
    # Check data information
    # print(df_train.shape)
    # print(df_train.info())
    # print(df_train.describe())
    # print(df_train.isnull().sum().head(10))
    # print(df_test.shape)
    # print(df_test.isnull().sum().head(10))

    # print(df_train['SalePrice'].describe())
    sns.distplot(df_train['SalePrice'])
    # plt.show()
    # skewness and kurtosis
    # print("Skewness: %f" % df_train['SalePrice'].skew())
    # print("Kurtosis: %f" % df_train['SalePrice'].kurt())

    # scatter plot grlivarea/saleprice
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    # fig = plt.figure()
    # plt.scatter(data[var], data['SalePrice'])
    # plt.xlabel(var)
    # plt.ylabel('SalePrice')
    # plt.ylim(0,800000)
    # plt.show()

    # scatter plot totalbsmtsf/saleprice
    var = 'TotalBsmtSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

    # box plot overallqual/saleprice
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)

    var = 'YearBuilt'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)

    # correlation matrix
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat)
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.show()
    # saleprice correlation matrix
    k = 10  # num of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    print(cols)
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, \
                     yticklabels=cols.values, xticklabels=cols.values)

    # scatterplot
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols])
    plt.show()

def missingCheck(df_train):
    # missing data
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))
    return missing_data

def GBR(X, y, Xtest):
    X_train, X_test, y_train, y_test \
        = train_test_split(np.array(X), np.array(y), \
                           test_size=0.3, random_state=42)

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 3,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    ytest = clf.predict(Xtest)
    out = pd.DataFrame({
        'Id': df_test['Id'], 'SalePrice': ytest})
    out.to_csv('./data/RFout4.csv', index=False)

def RF(X, y, Xtest):
    regr = RandomForestRegressor(max_depth=30, random_state=0)
    # X = df_train[cols].fillna(0)
    # X = StandardScaler().fit_transform(X)
    # y = df_train['SalePrice']
    # y = df_train['SalePrice'].apply(np.log)
    regr.fit(X, y)
    # print(mean_squared_error(y, pd.DataFrame(regr.predict(X)).apply(np.log)))
    print(mean_squared_error(y, regr.predict(X)))
    # print(regr.feature_importances_)
    # Xtest = df_test[cols].fillna(0)
    ytest = regr.predict(Xtest)
    out = pd.DataFrame({
        'Id': df_test['Id'], 'SalePrice': ytest})
    out.to_csv('./data/RFout3.csv', index=False)

def XGBR(X, y, Xtest):
    '''
    :param X: Train file data
    :param y: Train file SalePrice
    :param Xtest: Test file data
    :return:
    '''
    X_train, X_test, y_train, y_test \
        = train_test_split(np.array(X), np.array(y), \
                           test_size=0.3, random_state=42)
    xgb1 = XGBRegressor(base_score = 0.5,
                        booster='gbtree',
                        colsample_bylevel = 1,
                        colsample_bytree = 1,
                        gamma = 0,
                        learning_rate = 0.01,
                        max_delta_step = 3,
                        max_depth = 3,
                        min_child_weight=1,
                        missing=None,
                        n_estimators=100,
                        nthread=-1,
                        objective='reg:linear',
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=2018,
                        silent=True,
                        subsample=1)
    # -----Grid Search----
    # param_test = {
    #     'max_depth': [1, 2, 3, 4, 5, 6],
    #     'min_child_weight': [1, 2, 3, 4, 5, 6]
    # }
    # gsearch = GridSearchCV(estimator=xgb1,
    #                        param_grid=param_test,
    #                        scoring='mean_squared_error', n_jobs=4, iid=False, cv=5)
    #
    # train_model4 = gsearch.fit(np.array(X_train), np.array(y_train))
    # print("best para: ", gsearch.best_params_)
    # print("best score: ", gsearch.best_score_)
    # pred4 = train_model4.predict(X_test)
    # print(mean_squared_error(y, pred4))

    xgb1.fit(X_train,y_train,eval_metric='rmse')
    y_pre = xgb1.predict(X_test)
    print("r2_score: %.4f" % metrics.r2_score(y_pre, y_test))

    ytest = xgb1.predict(Xtest)
    out = pd.DataFrame({
        'Id': df_test['Id'], 'SalePrice': ytest})
    out.to_csv('./data/RFout5.csv', index=False)

def facAnalysis(data):
    # print("Neighborhood Information")
    # print(df_train['Neighborhood'].value_counts())
    # print(df_train['Neighborhood'].unique())

    '''
    Take Neighborhood as an example
    :param data:
    :return:
    '''
    var = df_train['Neighborhood'].unique()
    med = pd.DataFrame({
        'type': var, 'value': 0})
    # print(df_train.loc[df_train.loc[df_train['Neighborhood']==var[1]].index,'SalePrice'].median())
    for i in range(0, len(med['type'])):
        med.loc[i,'value'] = df_train.loc[df_train.loc[df_train['Neighborhood'] == var[i]].index, 'SalePrice'].mean()


    avg = med['value'].mean()
    avg = [avg]*len(med)

    med = med.sort_values(by = 'value')
    # med1.reset_index(inplace=True)
    med.index = range(len(med))
    # med1 = med1.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(12,9))
    # plt.figure(figsize=(12,9))
    ax.bar(range(len(med)),med['value'])
    ax.plot(range(len(med)), avg, linestyle='--', color = 'red', linewidth = 3)
    # ax.legend(med1['type'])
    ax.set_xlabel('Neighborhood')
    ax.set_ylabel('Mean SalePrice')
    plt.xticks(range(len(med)), med['type'], rotation=45)
    plt.show()

if __name__ == "__main__":
    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")
    df_test['SalePrice'] = np.nan
    # facAnalysis(df_train)
    combi = pd.concat([df_train,df_test])
    print(combi.isnull().sum())
    combi['PoolQC'] = combi['PoolQC'].map({np.nan:0,'Fa':1,'Gd':2,'Ex':3})
    print(combi['PoolQC'].value_counts())

    X_train = combi.loc[combi['SalePrice'].isin([np.nan]) == False]
    X_test = combi.loc[combi['SalePrice'].isin([np.nan]) == True]
    '''# ------feature analysis------'''
    # featureAnalysis(df_train, df_test)
    '''# ------check missing data------'''
    missing_data1 = missingCheck(df_train)
    # df_train = df_train.drop((missing_data1[missing_data1['Total'] > 1]).index, 1)
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    missing_data2 = missingCheck(df_test)
    # df_test = df_test.drop((missing_data2[missing_data2['Total'] > 1]).index, 1)
    df_test = df_test.drop(df_test.loc[df_test['Electrical'].isnull()].index)
    # print(df_train.isnull().sum().max())
    '''------# standardizing data------'''
    saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
    # print(saleprice_scaled)
    low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
    high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
    # print('outer range (low) of the distribution:')
    # print(low_range)
    # print('\nouter range (high) of the distribution: ')
    # print(high_range)
    # print(type(high_range))
    # print(saleprice_scaled.head())

    var = 'GrLivArea'

    # data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    # data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    # plt.show()
    # ------deleting points------
    print(df_train.sort_values(by='GrLivArea', ascending=False)[:2])
    df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
    # print(df_train['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea'])
    # print(df_train.columns)
    # bivariate analysis saleprice/grlivarea
    # var = 'TotalBsmtSF'
    # data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    # data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

    # sns.distplot(df_train['SalePrice'], fit=norm)
    # plt.show()
    # fig = plt.figure()
    # res = stats.probplot(df_train['SalePrice'], plot=plt)
    # # applying log transformation
    # df_train['SalePrice'] = np.log(df_train['SalePrice'])
    # # transformed histogram and normal probability plot
    # sns.distplot(df_train['SalePrice'], fit=norm);
    # fig = plt.figure()
    # res = stats.probplot(df_train['SalePrice'], plot=plt)
    #
    # plt.show()

    # #############################################################################
    # -----Choosing features-----
    featureCols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
       'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1',
       'LotFrontage', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath',
       'LotArea', 'BsmtFullBath', 'BsmtUnfSF', 'BedroomAbvGr', 'ScreenPorch']
    X = df_train[featureCols].fillna(0)
    X = StandardScaler().fit_transform(X)
    y = df_train['SalePrice']
    Xtest = df_test[featureCols].fillna(0)
    Xtest = StandardScaler().fit_transform(Xtest)

    # #############################################################################
    # -----RF regression model-----
    # RF(X, y, Xtest)

    # #############################################################################
    # -----Fit Gradient Boosting regression model-----
    # GBR(X, y, Xtest)

    # #############################################################################
    # -----Fit XGBoost regression model-----
    # XGBR(X, y, Xtest)








