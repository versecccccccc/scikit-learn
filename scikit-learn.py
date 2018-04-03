import numpy as np 
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn import neighbors  
from sklearn import svm 
from sklearn.naive_bayes import MultinomialNB  
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
import datetime
import time

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#data_preprocessing

#scale
def my_SCALE(X):
    ask_scaled = input("scale or not: ") 
    if ask_scaled == 'yes':
        X_scaled = preprocessing.scale(X)
        return X_scaled
    else:
        return X

#MinMaxScaler
def my_MMS(X):
    ask_mms = input("MinMaxScaler or not: ") 
    if ask_mms == 'yes':
        min_max_scaler = preprocessing.MinMaxScaler() 
        X_train_minmax = min_max_scaler.fit_transform(X)  
        return X_train_minmax
    else:
        return X

#MaxAbsScaler
def my_MAS(X):
    ask_mas = input("MaxAbsScaler or not: ") 
    if ask_mas == 'yes':
        max_abs_scaler = preprocessing.MaxAbsScaler()
        X_train_maxabs = max_abs_scaler.fit_transform(X)
        return X_train_maxabs
    else:
        return X

# Normalizer
def my_NM(X):
    ask_nm = input("Normalizer or not: ") 
    if ask_nm == 'yes':
        normalizer=preprocessing.Normalizer().fit(X)
        X_train_nominizer = normalizer.transform(X)
        return X_train_nominizer
    else:
        return X

#Binary
def my_BI(X):
    ask_bi = input("Binary or not: ") 
    if ask_bi == 'yes':
        binarizer = preprocessing.Binarizer(threshold=0)
        X_BI = binarizer.transform(X)
        return X_BI
    else:
        return X

#pca

def my_PCA(X):
    ask_pca = input("pca or not: ")
    if ask_pca == 'yes':
        n_com=2
        pca = PCA(n_components = n_com)
        X_pca = pca.fit_transform(X)
        return X_pca
    else:
        return X

def data_process(X):
    X_PCA = my_PCA(X)
    X_SCALE = my_SCALE(X_PCA)
    X_MMS = my_MMS(X_SCALE)
    X_MAS = my_MAS(X_MMS)
    X_NM = my_NM(X_MAS)
    X_BI = my_BI(X_NM)   
    return X_BI

def read_num_int(user_put):
    list = []
    length = len(user_put)
    s = user_put[1:length-1].split(',')
    #print(s)
    if user_put[0:1] == '(':
        begin = int(s[0])
        end = int(s[1])
        dis = int(s[2])
        lenth = end - begin + 1
        for i in range(0, lenth):
            list.append(begin)
            #print(list[i])
            begin += dis
    else: 
        str = user_put[1:length-1]
        r = len(str.split(","))
        for i in range(0, r):
            list.append(int(str.split(",")[i]))
            #print(list[i])
    return list

def read_num_float(user_put):
    list = []
    length = len(user_put)
    s = user_put[1:length-1].split(',')
    #print(s)
    if user_put[0:1] == '(':
        begin = float(s[0])
        end = float(s[1])
        dis = float(s[2])
        lenth = int((end - begin)/dis) + 1
        for i in range(0, lenth):
            list.append(begin)
            #print(list[i])
            begin += dis
    else: 
        str = user_put[1:length-1]
        r = len(str.split(","))
        for i in range(0, r):
            list.append(float(str.split(",")[i]))
            #print(list[i])
    return list

def read_str(user_put):
    lenth = len(user_put.split(","))
    #print(user_put)
    list = []
    for i in range(0, lenth):
        list.append(user_put.split(",")[i])
        #print(list[i])
    return list

#estimator
#n_iter_search = 20

#Random Forest
def RFC():        
    clf=RandomForestClassifier()
    print("RFC_user_input: ")
    user_input = input("max_depth|min_samples_split|min_samples_leaf|bootstrap|criterion: ")
    u_p = user_input.split('|')
    param_dist={"max_depth":read_num_int(u_p[0]),
            #"max_features":[1,5,10],
            "min_samples_split":read_num_int(u_p[1]),#read_num(user_input),
            "min_samples_leaf":read_num_int(u_p[2]),
            "bootstrap":[True,False],
            "criterion":read_str(u_p[3]) #['gini','entropy']
            }
    n_iter_search = 5
    start_rfc_r = time.time()
    predictor_r = RandomizedSearchCV(clf, param_distributions = param_dist, n_iter = n_iter_search)
    end_rfc_r = time.time()
    runtime_r = end_rfc_r - start_rfc_r
    start_rfc_g = time.time()
    predictor_g = GridSearchCV(clf, param_grid = param_dist)
    end_rfc_g = time.time()
    runtime_g = end_rfc_g - start_rfc_g
    return (predictor_r,predictor_g,runtime_r, runtime_g)
    #params = {'min_samples_leaf': range(1, 10), 'min_samples_split': range(2, 8)}
    #predictor = GridSearchCV(clf, params, n_jobs=-1)

#Logistic Regression
def LR():
    clf = LogisticRegression()
    print("LR_user_input: ")
    user_input = input("C: ")
    u_p = user_input.split('|')
    params = {'C':read_num_int(u_p[0])}#[1, 2, 4] 
    n_iter_search = 5
    start_lr_r = time.time()
    predictor_r = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    end_lr_r = time.time()
    runtime_r = end_lr_r - start_lr_r
    start_lr_g = time.time()
    predictor_g = GridSearchCV(clf, param_grid = params)
    end_lr_g = time.time()
    runtime_g = end_lr_g - start_lr_g
    return (predictor_r,predictor_g,runtime_r, runtime_g)
    #predictor = GridSearchCV(clf, params, n_jobs=-1)
    return predictor

#Gradboost
def GBC():
    clf = GradientBoostingClassifier()
    print("GBC_user_input: ")
    user_input = input("min_samples_leaf|min_samples_split: ")
    u_p = user_input.split('|')
    params = {'min_samples_leaf': read_num_int(u_p[0]), 'min_samples_split': read_num_int(u_p[1])}
    #params = {'min_samples_leaf': range(8, 10), 'min_samples_split': range(2, 8)}
    #n_iter_search = 5
    #predictor = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    n_iter_search = 5
    start_gbc_r = time.time()
    predictor_r = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    end_gbc_r = time.time()
    runtime_r = end_gbc_r - start_gbc_r
    start_gbc_g = time.time()
    predictor_g = GridSearchCV(clf, param_grid = params)
    end_gbc_g = time.time()
    runtime_g = end_gbc_g - start_gbc_g
    return (predictor_r,predictor_g,runtime_r, runtime_g)
    #predictor = GridSearchCV(clf, params, n_jobs=-1)
    return predictor

#Naive Bayes
def MNB():
    clf = MultinomialNB()
    print("MNB_user_input: ")
    user_input = input("alpha: ")
    u_p = user_input.split('|')
    params = {'alpha': read_num_float(u_p[0])}
    #params = {'alpha': [1.0, 2.0]}
    #n_iter_search = 5
    #predictor = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    n_iter_search = 5
    start_mnb_r = time.time()
    predictor_r = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    end_mnb_r = time.time()
    runtime_r = end_mnb_r - start_mnb_r
    start_mnb_g = time.time()
    predictor_g = GridSearchCV(clf, param_grid = params)
    end_mnb_g = time.time()
    runtime_g = end_mnb_g - start_mnb_g
    return (predictor_r,predictor_g,runtime_r, runtime_g)
    #predictor = GridSearchCV(clf, params, n_jobs=-1)
    return predictor

#SVM
def SVM():
    clf = svm.SVC() 
    print("SVM_user_input: ")
    user_input = input("kernel|C|gamma: ")
    u_p = user_input.split('|')
    params = {'kernel':read_str(u_p[0]), 'C':read_num_int(u_p[1]), 'gamma':read_num_float(u_p[2])}
    #params = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':}
    #n_iter_search = 5
    #predictor = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    n_iter_search = 5
    start_svm_r = time.time()
    predictor_r = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    end_svm_r = time.time()
    runtime_r = end_svm_r - start_svm_r
    start_svm_g = time.time()
    predictor_g = GridSearchCV(clf, param_grid = params)
    end_svm_g = time.time()
    runtime_g = end_svm_g - start_svm_g
    return (predictor_r,predictor_g,runtime_r, runtime_g)
    #predictor = GridSearchCV(clf, params, n_jobs=-1)
    return predictor

#KNN
def KNN():
    clf = neighbors.KNeighborsClassifier()
    print("KNN_user_input: ")
    user_input = input("n_neighbors|leaf_size: ")
    u_p = user_input.split('|')
    params = {'n_neighbors':read_num_int(u_p[0]), 'leaf_size': read_num_int(u_p[1])}
    #params = {'n_neighbors':range(3,10), 'leaf_size': range(20,40)}
    #n_iter_search = 5
    #predictor = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    n_iter_search = 5
    start_knn_r = time.time()
    predictor_r = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    end_knn_r = time.time()
    runtime_r = end_knn_r - start_knn_r
    start_knn_g = time.time()
    predictor_g = GridSearchCV(clf, param_grid = params)
    end_knn_g = time.time()
    runtime_g = end_knn_g - start_knn_g
    return (predictor_r,predictor_g,runtime_r, runtime_g)
    #predictor = GridSearchCV(clf, params, n_jobs=-1)
    return predictor

# Decision Tree
def TDC():
    clf = tree.DecisionTreeClassifier() 
    #print(clf.get_params())
    print("TDC_user_input: ")
    user_input = input("min_samples_leaf: ")
    u_p = user_input.split('|')
    params = {'min_samples_leaf': read_num_int(u_p[0])}
    #params = {'min_samples_leaf': range(5, 10)}
    #n_iter_search = 5
    #predictor = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    n_iter_search = 5
    start_tdc_r = time.time()
    predictor_r = RandomizedSearchCV(clf, param_distributions = params, n_iter = n_iter_search)
    end_tdc_r = time.time()
    runtime_r = end_tdc_r - start_tdc_r
    start_tdc_g = time.time()
    predictor_g = GridSearchCV(clf, param_grid = params)
    end_tdc_g = time.time()
    runtime_g = end_tdc_g - start_tdc_g
    return (predictor_r,predictor_g,runtime_r, runtime_g)
    #predictor = GridSearchCV(clf, params, n_jobs=-1)
    return predictor
    
#print(y_test)
#predictor.predict_proba(X_train)

def getRecognitionRate(xtestData, ytestData):  
    testNum = len(xtestData)  
    rightNum = 0  
    for i in range(0, testNum):  
        if ytestData[i] == xtestData[i]:  
            rightNum += 1  
    return float(rightNum) / float(testNum)

def validation(X, Y, Z):
    choose = input("held-out or cross: ")
    if choose == "cross":
        score1 = cross_val_score(X, Y, Z).mean()
        return score1
    else:
        score2 = X.score(Y, Z)
        return score2

def finalPredictor():
    #preprocessing
    print("For X_train data: ")
    X_final_train = data_process(X_train)    
    #print(X_final_train)
    print()
    
    print("For X_test data: ")
    X_final_test = data_process(X_test)    
    print()
    
    # get predictor
    (clf_RFC_R,clf_RFC_G,runtime_rfc_R,runtime_rfc_G) = RFC()
    (clf_LR_R,clf_LR_G,runtime_lr_R,runtime_lr_G) = LR()
    (clf_GBC_R,clf_GBC_G,runtime_gbc_R,runtime_gbc_G) = GBC()
    (clf_MNB_R,clf_MNB_G,runtime_mnb_R,runtime_mnb_G) = MNB()
    (clf_SVM_R,clf_SVM_G,runtime_svm_R,runtime_svm_G) = SVM()
    (clf_KNN_R,clf_KNN_G,runtime_knn_R,runtime_knn_G) = KNN()
    (clf_TDC_R,clf_TDC_G,runtime_tdc_R,runtime_tdc_G) = TDC()
    #clf_LR = LR()
    #clf_GBC = GBC()
    #clf_MNB = MNB()
    #clf_SVM = SVM()
    #clf_KNN = KNN()
    #clf_TDC = TDC()
    
    # input
    print("RFC validation: ")
    clf_RFC_R.fit(X_train, y_train)
    #RFC_score_R = validation(clf_RFC_R, X_final_train, y_train)
    RFC_score_R = clf_RFC_R.best_score_
    clf_RFC_G.fit(X_train, y_train)
    RFC_score_G = clf_RFC_G.best_score_
    #RFC_score_G = validation(clf_RFC_G, X_final_train, y_train)
    print("LR validation: ")
    clf_LR_R.fit(X_train, y_train)
    LR_score_R = clf_LR_R.best_score_
    clf_LR_G.fit(X_train, y_train)
    LR_score_G = clf_LR_G.best_score_
    print("GBC validation: ")
    #clf_GBC.fit(X_train, y_train)
    #GBC_score = validation(clf_GBC, X_final_train, y_train)
    clf_GBC_R.fit(X_train, y_train)
    GBC_score_R = clf_GBC_R.best_score_
    clf_GBC_G.fit(X_train, y_train)
    GBC_score_G = clf_GBC_G.best_score_
    print("MNB validation: ")
    #clf_MNB.fit(X_train, y_train)
    #MNB_score = validation(clf_MNB, X_final_train, y_train)
    clf_MNB_R.fit(X_train, y_train)
    MNB_score_R = clf_MNB_R.best_score_
    clf_MNB_G.fit(X_train, y_train)
    MNB_score_G = clf_MNB_G.best_score_
    print("SVM validation: ")
    #clf_SVM.fit(X_train, y_train)
    #SVM_score = validation(clf_SVM, X_final_train, y_train)
    clf_SVM_R.fit(X_train, y_train)
    SVM_score_R = clf_SVM_R.best_score_
    clf_SVM_G.fit(X_train, y_train)
    SVM_score_G = clf_SVM_G.best_score_
    print("KNN validation: ")
    #clf_KNN.fit(X_train, y_train)
    #KNN_score = validation(clf_KNN, X_final_train, y_train)
    clf_KNN_R.fit(X_train, y_train)
    KNN_score_R = clf_KNN_R.best_score_
    clf_KNN_G.fit(X_train, y_train)
    KNN_score_G = clf_KNN_G.best_score_
    print("TDC validation: ")    
    #clf_TDC.fit(X_train, y_train)
    clf_TDC_R.fit(X_train, y_train)
    TDC_score_R = clf_TDC_R.best_score_
    clf_TDC_G.fit(X_train, y_train)
    TDC_score_G = clf_TDC_G.best_score_
    #print(clf_TDC.predict_proba(X_test)[0][0])
    #print(clf_TDC.best_score_)
    #print(clf_TDC.best_estimator_.min_samples_leaf)
    #print(clf_TDC.predict_proba(X_train))
    #TDC_score = validation(clf_TDC, X_final_train, y_train)
    print()
    
    #score   
    print('RFC score RandomSearch: ', RFC_score_R)
    print('RFC score GridSearch: ', RFC_score_G)
    print(runtime_rfc_R)
    print(runtime_rfc_G)
    #print('RFC score GridSearch: ', RFC_score_G)
    #print('LR  score: ', LR_score)
    print('LR score RandomSearch: ', LR_score_R)
    print('LR score GridSearch: ', LR_score_G)
    print(runtime_lr_R)
    print(runtime_lr_G)
    #print('GBC score: ', GBC_score)
    print('GBC score RandomSearch: ', GBC_score_R)
    print('GBC score GridSearch: ', GBC_score_G)
    print(runtime_gbc_R)
    print(runtime_gbc_G)
    #print('MNB score: ', MNB_score)
    print('MNB score RandomSearch: ', MNB_score_R)
    print('MNB score GridSearch: ', MNB_score_G)
    print(runtime_mnb_R)
    print(runtime_mnb_G)
    #print('SVM score: ', SVM_score)
    print('SVM score RandomSearch: ', SVM_score_R)
    print('SVM score GridSearch: ', SVM_score_G)
    print(runtime_svm_R)
    print(runtime_svm_G)
    #print('KNN score: ', KNN_score)
    print('KNN score RandomSearch: ', KNN_score_R)
    print('KNN score GridSearch: ', KNN_score_G)
    print(runtime_knn_R)
    print(runtime_knn_G)
    #print('TDC score: ', TDC_score)
    print('TDC score RandomSearch: ', TDC_score_R)
    print('TDC score GridSearch: ', TDC_score_G)
    print(runtime_tdc_R)
    print(runtime_tdc_G)
    print()
    
    #parameters
    print(clf_RFC_R.best_params_)
    print(clf_RFC_G.best_params_)
    print(clf_LR_R.best_params_)
    print(clf_LR_G.best_params_)
    print(clf_GBC_R.best_params_)
    print(clf_GBC_G.best_params_)
    print(clf_MNB_R.best_params_)
    print(clf_MNB_G.best_params_)
    print(clf_SVM_R.best_params_)
    print(clf_SVM_G.best_params_)
    print(clf_KNN_R.best_params_)
    print(clf_KNN_G.best_params_)
    print(clf_TDC_R.best_params_)
    print(clf_TDC_G.best_params_)
    
    #recognition rate
    print('RFC recognition rate: ', getRecognitionRate(clf_RFC_R.predict(X_final_test), y_test))
    print('RFC recognition rate: ', getRecognitionRate(clf_RFC_G.predict(X_final_test), y_test))
    print('LR recognition rate: ', getRecognitionRate(clf_LR_R.predict(X_final_test), y_test))
    print('LR recognition rate: ', getRecognitionRate(clf_LR_G.predict(X_final_test), y_test))
    print('GBC recognition rate: ', getRecognitionRate(clf_GBC_R.predict(X_final_test), y_test))
    print('GBC recognition rate: ', getRecognitionRate(clf_GBC_G.predict(X_final_test), y_test))
    print('MNB recognition rate: ', getRecognitionRate(clf_MNB_R.predict(X_final_test), y_test))
    print('MNB recognition rate: ', getRecognitionRate(clf_MNB_G.predict(X_final_test), y_test))
    print('SVM recognition rate: ', getRecognitionRate(clf_SVM_R.predict(X_final_test), y_test))
    print('SVM recognition rate: ', getRecognitionRate(clf_SVM_G.predict(X_final_test), y_test))
    print('KNN recognition rate: ', getRecognitionRate(clf_KNN_R.predict(X_final_test), y_test))
    print('KNN recognition rate: ', getRecognitionRate(clf_KNN_G.predict(X_final_test), y_test))
    print('KNN recognition rate: ', getRecognitionRate(clf_TDC_R.predict(X_final_test), y_test))
    print('KNN recognition rate: ', getRecognitionRate(clf_TDC_G.predict(X_final_test), y_test))
    '''print('GBC recognition rate: ', getRecognitionRate(clf_GBC.predict(X_final_test), y_test))
    print('MNB recognition rate: ', getRecognitionRate(clf_MNB.predict(X_final_test), y_test))
    print('SVM recognition rate: ', getRecognitionRate(clf_SVM.predict(X_final_test), y_test))
    print('KNN recognition rate: ', getRecognitionRate(clf_KNN.predict(X_final_test), y_test))
    print('TDC recognition rate: ', getRecognitionRate(clf_TDC.predict(X_final_test), y_test))
    print()'''

if __name__ == '__main__':  
    finalPredictor()
    print('The End.')

