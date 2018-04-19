# -*- coding: utf-8 -*-
'''
@author: Youngway
'''
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def creatFeature(train, test, features):
    combi = pd.concat([train, test])[features + ['Name', 'PassengerId', 'Survived']]
    combi['Embarked'] = combi['Embarked'].map({'S': 1, "Q": 2, "C": 3}).fillna(1)
    combi['Fare'] = combi['Fare'].fillna(method='bfill', axis=0).fillna("0")
    combi['Sex'] = combi['Sex'].map({'male': 0, 'female': 1})
    combi['Age'] = combi['Age'].fillna(0)
    combi['Pclass'] = combi['Pclass'].apply(lambda x: 10/np.exp(x))
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    combi['Title'] = combi.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    combi['Title'] = combi['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    combi['Title'] = combi['Title'].replace(['Mlle', 'Ms'], 'Miss')
    combi['Title'] = combi['Title'].replace('Mme', 'Mrs')
    combi['Title'] = combi['Title'].map(title_mapping)
    combi['Title'] = combi['Title'].fillna(0)
    print(combi['Pclass'].head())
    print(combi.head())
    print(combi.isnull().sum())
    return combi

def RF(X_train, Y_train):
    # Use gridsearch to fine tune RF
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    n_estimators = [200, 500]
    max_features = ['auto', 'sqrt']
    max_depth = [10, 30, 50, 100]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}
    model = RandomForestClassifier()
    kfold = KFold(n_splits=10, random_state=7)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    model = RandomForestClassifier(n_estimators=1000, max_features='auto', max_depth=100, min_samples_split=10,
                                   min_samples_leaf=1, bootstrap=True)
    model.fit(rescaledX, Y_train)
    return model

def SVC_model(X_train, Y_train):
    # Use gridsearch to fine tune SVM
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    model = SVC()
    kfold = KFold(n_splits=10, random_state=7)
    print("kfold:", kfold)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    model = SVC(decision_function_shape='ovr', C=10, cache_size=2000,\
              degree= 10, gamma= 0.1, kernel='rbf', tol = 0.0001, probability=True)
    model.fit(rescaledX, Y_train)
    return model

def xgb_model(dataset1):
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': 12
              }
    model = xgb.train(params, dataset1, num_boost_round=3000, evals=[(dataset1,'train')])
    return model

def xgb_classification(X_train, Y_train):
    model = XGBClassifier(
                learning_rate=0.1,
                n_estimators=1000,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=27)
    train_model = model.fit(X_train, Y_train)
    pred = train_model.predict(X_train)
    print("Accuracy for model: %.2f" % (accuracy_score(Y_train, pred) * 100))
    return  train_model


if __name__ == "__main__":
    train = pd.read_csv("D:\ying\Pycharm\ying\Titanic\data\\train.csv")
    # print(train.Survived)
    test = pd.read_csv("D:\ying\Pycharm\ying\Titanic\data\\test.csv")
    test['Survived'] = np.nan
    # features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    combi = creatFeature(train, test, features)
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title']
    train_sample = combi.loc[combi['Survived'].isin([np.nan]) == False]
    test_sample = combi.loc[combi['Survived'].isin([np.nan]) == True]

    # dataset1 = xgb.DMatrix(np.array(train_sample[features]), label=np.array(train_sample['Survived']))
    # dataset2 = xgb.DMatrix(np.array(test_sample[features]))
    # model = xgb_model(dataset1)
    # pre = model.predict(dataset2)
    # print(pre)
    X_train, X_test, y_train, y_test \
        = train_test_split(np.array(train_sample[features]), np.array(train_sample['Survived']),\
                           test_size=0.3, random_state=42)
    # Train the XGboost Model for Classification
    model1 = xgb.XGBClassifier()
    model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)

    train_model1 = model1.fit(X_train, y_train)
    train_model2 = model2.fit(X_train, y_train)

    # prediction and Classification Report
    from sklearn.metrics import classification_report

    pred1 = train_model1.predict(X_test)
    pred2 = train_model2.predict(X_test)

    print("Accuracy 1: ", accuracy_score(y_test, pred1))
    print("Accuracy 2: ", accuracy_score(y_test, pred2))
    # print('Model 1 XGboost Report %r' % (classification_report(y_test, pred1)))
    # print('Model 2 XGboost Report %r' % (classification_report(y_test, pred2)))

    # Let's do a little Gridsearch, Hyperparameter Tunning
    model3 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    train_model3 = model3.fit(X_train, y_train)
    pred3 = train_model3.predict(X_test)
    print("Accuracy for model 3: %.2f" % (accuracy_score(y_test, pred3) * 100))

    from sklearn.model_selection import GridSearchCV

    param_test = {
        'max_depth': [4,5,6],
        'min_child_weight': [4,5,6]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                   n_estimators=140,
                                                   max_depth=8,
                                                   min_child_weight=4,
                                                   gamma=0,
                                                   subsample=0.8,
                                                   colsample_bytree=0.8,
                                                   objective='binary:logistic',
                                                   nthread=5,
                                                   scale_pos_weight=1,
                                                   seed=27),
                           param_grid=param_test, scoring='accuracy', n_jobs=4, iid=False, cv=5)

    train_model4 = gsearch.fit(np.array(train_sample[features]), np.array(train_sample['Survived']))
    print("best para: ", gsearch.best_params_)
    print("best score: ", gsearch.best_score_)
    pred4 = train_model4.predict(X_test)
    # print(pred4)
    # print("Accuracy for model 4: %.2f" % (accuracy_score(y_test, pred4) * 100))
    # clf = train_model4.best_estimator_
    # clf.fit(np.array(train_sample[features]), np.array(train_sample['Survived']))
    pre = gsearch.predict(np.array(test_sample[features]))

    # train_model = xgb_classification(np.array(train_sample[features]), np.array(train_sample['Survived']))
    # pre = train_model.predict(np.array(test_sample[features]))
    # print("Accuracy for model 3: %.2f" % (accuracy_score(y_test, pred3) * 100))

    #
    # model = SVC_model(X_train, Y_train)
    # pre = model.predict(np.array(test_sample[features]))
    # print(pre)
    # c = [1,2,3,4,5,6,7,8,9,10,20,30,50,100]
    # g = [0.001,0.002,0.003,0.005,0.01,0.02,0.03,0.04,0.05,0.08,0.1,0.2,0.5,1.0]
    # svc_scores = []
    # for i in c:
    #     for j in g:
    #         model = SVC(C=i,gamma=j)
    #         scores = cross_val_score(model, np.array(train_sample[features]), np.array(train_sample['Survived']), cv = 10, scoring='accuracy')
    #         svc_scores.append([i,j,scores.mean()])
    # plt.figure()
    # num = len(c)* len(g)
    # svc_scores = pd.DataFrame(svc_scores)

    # print(num)
    # plt.plot(range(num), svc_scores)
    # plt.show()
    # Final
    # clf = SVC()
    # para ={'C': [1,2,5,10,16,32,64,128], 'gamma':[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]}
    # grs = GridSearchCV(clf, param_grid= para, cv=10, n_jobs=1, return_train_score=False)
    # grs.fit(np.array(train_sample[features]), np.array(train_sample['Survived']))
    # print("Best paremters:" + str(grs.best_params_))
    # gpd = pd.DataFrame(grs.cv_results_)
    # print(gpd.head())
    # print("Accuracy: {:.4f}".format(gpd['mean_test_score'][grs.best_index_]))
    # pre = grs.predict(np.array(test_sample[features]))
    out = pd.DataFrame({
        'PassengerId': test_sample['PassengerId'], 'Survived': pre})
    out.to_csv('./data/xgb_out_new5.csv', index=False, float_format='%1d')
