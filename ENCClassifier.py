import time
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import variation 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import scipy.spatial.distance as dist
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator

# ------------------------------------------------------------------

def Trimming(X):
    Xp = X
    top5 = []
    bottom5 = []
    for c in X.columns[(X.dtypes != object) & (X.dtypes != "category")]:
        # winsorization
        if X[c].nunique()>=50: 
            cut_off95 = np.percentile(X[c],95)
            cut_off05 = np.percentile(X[c],5)
        else:
            cut_off95 = 9999999999.
            cut_off05 =-9999999999.
        top5.append(cut_off95)
        bottom5.append(cut_off05)
        Xp[c] = np.where(X[c]>cut_off95,cut_off95,X[c])
        Xp[c] = np.where(X[c]<cut_off05,cut_off05,X[c])
    return Xp, top5, bottom5

def Calculate_Relevance(data, target, bins=10):
    Rel_main_df = pd.DataFrame()
    Rel_detail_df = pd.DataFrame()
    cols = data.columns
    for ivars in cols[~cols.isin([target])]:
        My_y = data[target]
        if (data[ivars].dtype.kind in 'biufc') and (len(np.unique(data[ivars]))>bins):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            #binned_x = np.digitize(data[ivars], bins=np.quantile(data[ivars],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
            d0 = pd.DataFrame({'x': binned_x, 'y': My_y})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': My_y})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['%'] = d['Events'] / d['N']
        d['Lift'] = d['%'] / My_y.mean()
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['Importance'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        # print("Information value of " + ivars + " is " + str(round(d['IV Part'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "Importance" : [d['Importance'].sum()]}, columns = ["Variable", "Importance"])
        Rel_main_df=pd.concat([Rel_main_df,temp], axis=0)
        Rel_detail_df=pd.concat([Rel_detail_df,d], axis=0)
            
    Rel_main_df = Rel_main_df.set_index('Variable')
    Rel_main_df["Importance"]=np.where(Rel_main_df["Importance"]>1000,1000,Rel_main_df["Importance"])

    return Rel_main_df, Rel_detail_df

# ------------------------------------------------------------------

def Centroid(X_train, y_train, class_value):
    X_train = X_train.loc[y_train==class_value]
    X = X_train.to_numpy()
    X_Center = X.mean(axis=0)
    X_Var = X.var(ddof=1,axis=0)
    X_Var = np.where(X_Var<=X_Center*0.05,X_Center*0.05,X_Var)
    X_Var = np.where(X_Var<=0.05,0.05,X_Var)
    return X_Center, X_Var

def Calc_Distance (df, X_Var, X_Center, Dist_type, Imp_type, Importance):
    
    Hom = 1. / X_Var

    if Imp_type == 1:
        ImpHom = Hom*(Importance['Importance'])
    elif Imp_type == 2:
        ImpHom = Hom*(1+Importance['Importance'])
    elif Imp_type == 3:
        ImpHom = Hom*((Importance['Importance'])**2)
    elif Imp_type == 4:
        ImpHom = Hom*((1+Importance['Importance'])**2)
    elif Imp_type == 5:
        ImpHom = Hom*(np.sqrt(Importance['Importance']))
    elif Imp_type == 6:
        ImpHom = Hom*(np.sqrt(1+Importance['Importance']))
    elif Imp_type == 7:
        ImpHom = Hom*(np.log(1+Importance['Importance']))
    else:
        ImpHom = Hom*((1+Importance['Importance'])**2)
    
    if Dist_type == 'L1':
        # L1: Manhattan style
        Dif = np.absolute(df - X_Center)
    elif Dist_type == 'L2':
        # L2: Euclidean style
        Dif = (df - X_Center)**2
    else:
        Dif = (df - X_Center)**2
    
    Distance = Dif.dot(ImpHom)
    Distance = np.where(Distance<0.000001,0.000001,Distance)
    
    Distance = pd.DataFrame(Distance)
    return Distance

# ------------------------------------------------------------------

def Calc_Performance(pred, pred_class, y):
    TN, FP, FN, TP = confusion_matrix(y, pred_class, labels=[0, 1]).ravel()
    if TP+FN>0:
        accuracy    = (TP+TN)/(TP+TN+FP+FN)
        sensitivity = TP / (TP+FN)
        specifity   = TN / (TN+FP)
        fpr, tpr, thresholds = roc_curve(y, pred)
        AUC = auc(fpr, tpr)
        GINI = 2 * AUC - 1
    else:
        accuracy    = 0
        sensitivity = 0
        specifity   = 0        
        fpr=0
        tpr=0
        thresholds = 0
        AUC = 0
        GINI = 0
    return TP, FP, TN, FN, accuracy, sensitivity, specifity, AUC, GINI

# ------------------------------------------------------------------

def ENC(X, y, Dist_type, X0_Center, X0_Var, X1_Center, X1_Var, 
         Importance, train_mode, cut_off, Imp_type, pred_min, pred_max):
    Distance0 = Calc_Distance(X, X0_Var, X0_Center, Dist_type, Imp_type, Importance )
    Distance1 = Calc_Distance(X, X1_Var, X1_Center, Dist_type, Imp_type, Importance )
    # conversion to probabilities (scores)
    DistanceSum = Distance0.sum(axis=1) + Distance1.sum(axis=1)
    Distance0 = Distance0.div(DistanceSum,axis=0)
    Distance1 = Distance1.div(DistanceSum,axis=0)
    pred_class_info = pd.concat([Distance0.sum(axis=1),Distance1.sum(axis=1)], axis=1)
        
    # NET Similarity to class ONE
    pred = pred_class_info.iloc[:,0] - pred_class_info.iloc[:,1]
    pred = (pred-pred_min)/(pred_max-pred_min)
    pred = pd.DataFrame(pred,columns=['score'])
    pred_class = np.full((len(pred_class_info),1), False, dtype=bool)

    if train_mode==True:
        pred_class[pred.nlargest(y.sum(), ['score']).index] = True
        cut_off = float(pred[pred_class==True].min())
    else:
        pred_class[pred>=cut_off] = True
    pred_class = np.array(pred_class[:,0],dtype=bool)
    
    TP, FP, TN, FN, accuracy, sensitivity, specifity, AUC, GINI = Calc_Performance(pred, pred_class, y)
    exp_results = pd.DataFrame(columns = ['dataset', 'Dist_type', 'TP', 'FP', 'TN', 'FN', 'accuracy', 'sensitivity', \
                                          'specifity', 'AUC', 'GINI', 'cut_off', 'Top10_Flag'])
    exp_results.loc[0,'TP'] = TP
    exp_results['FP'] = FP
    exp_results['TN'] = TN
    exp_results['FN'] = FN
    exp_results['accuracy'] = accuracy
    exp_results['sensitivity'] = sensitivity
    exp_results['specifity'] = specifity
    exp_results['AUC'] = AUC
    exp_results['GINI'] = GINI
    exp_results['cut_off'] = 0
    exp_results['Top10_Flag'] = 0
    exp_results['Dist_type'] = Dist_type
    exp_results['Imp_type'] = Imp_type
    
    return pred, pred_class, exp_results

# ------------------------------------------------------------------

def ENC_Tuning(X_train, y_train, My_results, Dist_type, Imp_type):
    # LEARNING PHASE STEP 1 - Importance
    #X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train,name='y')
    My_data = pd.concat([X_train,y_train],axis=1)
    Rel_main, Rel_detail = Calculate_Relevance(data=My_data, target='y')
    Importance = Rel_main.copy()
    #Most_important_columns = Rel_main.nlargest(math.floor(len(df.columns)*0.8),Importance).index
    tmp = Importance[Importance.Importance>=0.0]
    Most_important_columns = tmp.nlargest(10000,Importance).index
    # LEARNING PHASE STEP 2 - Centroids
    X0_Center, X0_Var = Centroid(X_train, y_train, 0)
    X1_Center, X1_Var = Centroid(X_train, y_train, 1)
    pred_min = 0
    pred_max = 1
    # LEARNING PHASE STEP 3: ENC
    pred, pred_class, exp_results = ENC(X_train,y_train, Dist_type, X0_Center, X0_Var, X1_Center, X1_Var, \
                                        Importance, True, 0, Imp_type, pred_min, pred_max)
    My_results = pd.concat([My_results,exp_results], ignore_index = True)
    pred_min = float(pred.min())
    pred_max = float(pred.max())
    
    return My_results, Importance, X0_Center, X0_Var, X1_Center, X1_Var, Rel_main, Rel_detail, Most_important_columns, pred_min, pred_max

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

class Performance:
    def __init__(self, pred, pred_class, y):
        TP, FP, TN, FN, accuracy, sensitivity, specifity, AUC, GINI = Calc_Performance(pred, pred_class, y)
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.accuracy = accuracy
        self.sensitivity = sensitivity
        self.specifity = specifity
        self.AUC = AUC
        self.GINI = GINI

# --------------

class ENCClassifier(BaseEstimator):
    def __init__(self, Dist_type="L2", Imp_type=4):
        self.Dist_type=Dist_type
        self.Imp_type = Imp_type
        
    def fit(self, X, y):
        self.X_train = pd.DataFrame(X)
        self.y_train = pd.Series(y)
        self.X_train, self.top5, self.bottom5 = Trimming(self.X_train)
        self.My_results = pd.DataFrame(columns = ['dataset', 'Dist_type', 'Imp_type', 'TP', 'FP', 'TN', 'FN', 'accuracy', 'sensitivity', 'specifity', 'AUC', 'GINI', 'cut_off', 'Top10_Flag'])
        self.Importance             = pd.DataFrame()
        self.X0_Center, self.X0_Var = pd.DataFrame(), pd.DataFrame()
        self.X1_Center, self.X1_Var = pd.DataFrame(), pd.DataFrame()
        self.Rel_main, self.Rel_detail, self.Most_important_columns   = pd.DataFrame(), pd.DataFrame(), []
        self.My_results, self.Importance,                              \
        self.X0_Center, self.X0_Var, self.X1_Center, self.X1_Var,      \
        self.Rel_main, self.Rel_detail, self.Most_important_columns,   \
        self.pred_min, self.pred_max                                   \
        = ENC_Tuning(self.X_train, self.y_train, self.My_results, self.Dist_type, self.Imp_type)
        self.My_results.AUC = self.My_results.AUC.astype(float)
        self.best_params = self.My_results.loc[0,['Dist_type','cut_off','Imp_type']]
        self.best_cut_off = float(self.My_results.loc[0,['cut_off']])
        #print('Best Params=',self.best_params)
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.accuracy = 0
        self.sensitivity = 0
        self.specifity = 0
        self.AUC = 0
        return
    
# --------------

    def __predictions(self, X):
        temp = ENC(X, y = np.zeros(X.shape[0]), 
                   Dist_type = self.best_params.Dist_type,
                   X0_Center=self.X0_Center, X0_Var=self.X0_Var, 
                   X1_Center=self.X1_Center, X1_Var=self.X1_Var, 
                   Importance=self.Importance, 
                   train_mode=False,
                   cut_off=self.best_cut_off,
                   Imp_type=self.best_params.Imp_type,
                   pred_min = self.pred_min,
                   pred_max = self.pred_max
                  )
        return temp

    def predict(self, X):
        return pd.DataFrame(self.__predictions(X)[1])

    def predict_proba(self, X):
        return self.__predictions(X)[0]


