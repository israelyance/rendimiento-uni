#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:20:46 2019

@author: israelyance
"""

# Función 
##########################################

def resumen_table(X, y):
    
    '''
    Función para obtener un dataframe con los indicadores de
    aciertos y gini  de cada modelo de entrenamiento. 
    '''
    
    # Importando librerias
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    
    # Dividir el dataset en conjunto de entrenamiento y conjunto de testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    
    columns = ['logit', 'naive_bayes', 'svr', 'svc', 'tree_cl', 'tree_reg',
                     'rf_cl', 'rf_reg', 'xgb_cl', 'xgb_reg']
    index = ['train_aciertos', 'train_gini', 'test_aciertos', 'test_gini']
    resumen = pd.DataFrame(index = index, columns = columns)
    
    
    
    # LOGISTIC
    ################################################
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression(random_state = 0)
    logistic.fit(X_train, y_train)
    # Precisión
    resumen.loc['train_aciertos','logit'] = logistic.score(X_train, y_train) # Train
    resumen.loc['test_aciertos','logit'] = logistic.score(X_test, y_test) # Test
    # Poder de discriminación: Gini
    from sklearn.metrics import roc_auc_score
    resumen.loc['train_gini','logit'] = 2*roc_auc_score(y_train, logistic.predict_proba(X_train)[:, 1]) - 1 # Train
    resumen.loc['test_gini','logit'] = 2*roc_auc_score(y_test, logistic.predict_proba(X_test)[:, 1]) -1 # Test
    
    
    # NAIVE BAYES
    ################################################
    from sklearn.naive_bayes import MultinomialNB
    nbayes = MultinomialNB()
    nbayes.fit(X_train, y_train)
    # Precisión
    resumen.loc['train_aciertos','naive_bayes'] = nbayes.score(X_train, y_train) # Train
    resumen.loc['test_aciertos','naive_bayes'] = nbayes.score(X_test, y_test) # Test
    # Discriminación
    resumen.loc['train_gini','naive_bayes'] = 2*roc_auc_score(y_train, nbayes.predict_proba(X_train)[:, 1]) - 1 # Train
    resumen.loc['test_gini','naive_bayes']  = 2*roc_auc_score(y_test, nbayes.predict_proba(X_test)[:, 1]) -1 # Test
    
    
    
    
    # SVM Regresión
    ################################################
    from sklearn.svm import SVR # Librería
    svr = SVR() # Objeto
    svr.fit(X_train, y_train) # Entrenamiento
    # Precisión
    resumen.loc['train_aciertos','svr'] = svr.score(X_train, y_train) # Train
    resumen.loc['test_aciertos','svr'] = svr.score(X_test, y_test) # Test
    # Discriminación
    resumen.loc['train_gini','svr'] = 2*roc_auc_score(y_train, svr.predict(X_train)) - 1 # Train
    resumen.loc['test_gini','svr']  = 2*roc_auc_score(y_test, svr.predict(X_test)) -1 # Test
    
    
    
    # SVM Clasificación
    ################################################
    from sklearn.svm import SVC
    svc = SVC() # Objeto
    svc.fit(X_train, y_train) # Entrenamiento
    # Precisión
    resumen.loc['train_aciertos','svc'] = svc.score(X_train, y_train) # Train
    resumen.loc['test_aciertos','svc'] = svc.score(X_test, y_test) # Test
    # Discriminación
    resumen.loc['train_gini','svc'] = 2*roc_auc_score(y_train, svc.predict(X_train)) - 1 # Train
    resumen.loc['test_gini','svc']  = 2*roc_auc_score(y_test, svc.predict(X_test)) -1 # Test
    
    
    
    # CLASIFICATION TREE
    ################################################
    from sklearn.tree import DecisionTreeClassifier # Libreria
    arbol_cl = DecisionTreeClassifier(max_depth = 10) # Objeto
    arbol_cl.fit(X_train, y_train) # Entrenamiento
    # Precisión
    resumen.loc['train_aciertos','tree_cl'] = arbol_cl.score(X_train, y_train)
    resumen.loc['test_aciertos','tree_cl'] = arbol_cl.score(X_test, y_test)
    # Discriminación
    resumen.loc['train_gini','tree_cl'] = 2*roc_auc_score(y_train, arbol_cl.predict_proba(X_train)[:, 1]) - 1 # Train
    resumen.loc['test_gini','tree_cl']  = 2*roc_auc_score(y_test, arbol_cl.predict_proba(X_test)[:, 1]) - 1 # Test
    
    
    
    # REGRESSION TREE
    ################################################
    from sklearn.tree import DecisionTreeRegressor # Libreria
    arbol_reg = DecisionTreeRegressor(max_depth = 10) # Objeto a optimizar
    arbol_reg.fit(X_train, y_train) # Entrenamiento
    # Precisión
    resumen.loc['train_aciertos','tree_reg'] = arbol_reg.score(X_train, y_train)
    resumen.loc['test_aciertos','tree_reg'] = arbol_reg.score(X_test, y_test)
    # Discriminación
    resumen.loc['train_gini','tree_reg'] = 2*roc_auc_score(y_train, arbol_reg.predict(X_train)) - 1 # Train
    resumen.loc['test_gini','tree_reg']  = 2*roc_auc_score(y_test, arbol_reg.predict(X_test)) - 1 # Test
    
    
    
    # RANDOM FOREST CLASSIFIER
    ################################################
    from sklearn.ensemble import RandomForestClassifier # Librería
    rf_cl = RandomForestClassifier(max_depth = 10, n_estimators = 10, random_state = 4) # Objeto
    rf_cl.fit(X_train, y_train) # Entrenamiento
    # Precisión
    resumen.loc['train_aciertos','rf_cl'] = rf_cl.score(X_train, y_train) # Train
    resumen.loc['test_aciertos','rf_cl'] = rf_cl.score(X_test, y_test) # Test
    # Discriminación
    resumen.loc['train_gini','rf_cl'] = 2*roc_auc_score(y_train, rf_cl.predict(X_train)) - 1 # Train
    resumen.loc['test_gini','rf_cl']  = 2*roc_auc_score(y_test, rf_cl.predict(X_test)) -1 # Test
    
    
    
    # RANDOM FOREST REGRESSION
    ################################################
    from sklearn.ensemble import RandomForestRegressor # Librería
    rf_reg = RandomForestRegressor(max_depth = 10, n_estimators = 10, random_state = 4) # Objeto
    rf_reg.fit(X_train, y_train) # Entrenamiento
    # Precisión
    resumen.loc['train_aciertos','rf_reg'] = rf_reg.score(X_train, y_train) # Train
    resumen.loc['test_aciertos','rf_reg'] = rf_reg.score(X_test, y_test) # Test
    # Discriminación
    resumen.loc['train_gini','rf_reg'] = 2*roc_auc_score(y_train, rf_reg.predict(X_train)) - 1 # Train
    resumen.loc['test_gini','rf_reg']  = 2*roc_auc_score(y_test, rf_reg.predict(X_test)) -1 # Test
    
    
    
    # XGBOOST CLASSIFIER
    ################################################
    import xgboost as xgb
    xgb_cl = xgb.XGBClassifier(objective = 'binary:logistic', max_depth = 10,
                               n_estimators = 10, seed = 4) # Objeto
    xgb_cl.fit(X_train, y_train) # Entrenamiento
    # Precisión
    resumen.loc['train_aciertos','xgb_cl'] = xgb_cl.score(X_train, y_train) # Train
    resumen.loc['test_aciertos','xgb_cl'] = xgb_cl.score(X_test, y_test) # Test
    # Discriminación
    resumen.loc['train_gini','xgb_cl'] = 2*roc_auc_score(y_train, xgb_cl.predict(X_train)) - 1 # Train
    resumen.loc['test_gini','xgb_cl']  = 2*roc_auc_score(y_test, xgb_cl.predict(X_test)) -1 # Test
    
    
    
    
    # XGBOOST REGRESSOR
    ################################################
    xgb_reg = xgb.XGBRegressor(objective = 'binary:logistic', max_depth = 10,
                               n_estimators = 10, seed = 4) # Objeto
    xgb_reg.fit(X_train, y_train) # Entrenamiento
    # Precisión
    resumen.loc['train_aciertos','xgb_reg'] = xgb_reg.score(X_train, y_train) # Train
    resumen.loc['test_aciertos','xgb_reg'] = xgb_reg.score(X_test, y_test) # Test
    # Discriminación
    resumen.loc['train_gini','xgb_reg'] = 2*roc_auc_score(y_train, xgb_reg.predict(X_train)) - 1 # Train
    resumen.loc['test_gini','xgb_reg']  = 2*roc_auc_score(y_test, xgb_reg.predict(X_test)) -1 # Test



    return resumen




