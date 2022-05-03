import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from  ngboost  import  NGBClassifier
from xgboost import XGBClassifier
#from xgboost import XGBClassifier as XGBC
# from lightgbm import LGBMClassifier
# import catboost as cb
from  ngboost.distns  import  k_categorical ,  Bernoulli
import smote_variants
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from collections import Counter
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import imblearn
from xgboost import XGBRegressor as XRandomForestRegressGBR
from PIL import Image
from decimal import Decimal
from sklearn import tree
from PIL import Image
import warnings
st.title("Coronary stenosis prediction system")



HBA1C= st.sidebar.slider('HBA1C',1,50)
HBA1C=HBA1C
pulsepressure= st.sidebar.slider('pulsepressure',1,200)
pulsepressure=pulsepressure
#pulsepressure = st.sidebar.number_input("pulsepressure", 1, 200)
#TNI= st.sidebar.slider('TNI',0.000,1.000)
#TNI=TNI

TNI = st.sidebar.number_input("TNI", 0.01, 1.00)
hypertension_list=['no','Class 1','Class 2','Class 3']
hypertension_name = st.sidebar.selectbox(
    "Hypertension",
    hypertension_list
)
if hypertension_name=='no':
    hypertension_num =0;
    if hypertension_name == 'Class 1':
        hypertension_num = 1;
        if hypertension_name == 'Class 2':
            hypertension_num = 2;
else:
    hypertension_num =3
    
FT4= st.sidebar.slider('FT4',0,40)
FT4=FT4
AST= st.sidebar.slider('AST',0,2000)
AST=AST
TSH= st.sidebar.slider('TSH',0,30)
TSH=TSH
AGES= st.sidebar.slider('AGES',0,100)
AGES=AGES
UA= st.sidebar.slider('UA',0,30)
UA=UA
Diseasetime= st.sidebar.slider('Diseasetime(Mouths)',0,60)
Diseasetime=Diseasetime
FT3= st.sidebar.slider('FT3',0,10)
FT3=FT3
LVDD= st.sidebar.slider('LVDD',0,100)
LVDD=LVDD
RV= st.sidebar.slider('RV',0,100)
RV=RV
TG= st.sidebar.slider('TG',0,50)
TG=TG
BMI= st.sidebar.slider('BMI',10,50)
BMI=BMI
TBIL= st.sidebar.slider('TBIL',0,100)
TBIL=TBIL
GLU= st.sidebar.slider('GLU',0,50)
GLU=GLU
EF= st.sidebar.slider('EF',0,100)
EF=EF


MR_list=['yes','no']
MR_name = st.sidebar.selectbox(
    "MR",
    MR_list
)
if MR_name=='yes':
    MR_num =1
else:
    MR_num =0

WMA_list=['yes','no']
WMA_name = st.sidebar.selectbox(
    "WMA",
    WMA_list
)
if WMA_name=='yes':
    WMA_num =1
else:
    WMA_num =0




data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')
data3 = pd.read_csv('data3.csv')



# 缺失值填补data1

# 按列计算缺失值的函数
def missing_values_table(data):
    # 计算总的缺失值
    mis_val = data.isnull().sum()

    # 计算缺失值的百分比
    mis_val_percent = 100 * data.isnull().sum() / len(data)

    # 把结果制成表格
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # 对列重命名，第一列：Missing Values，第二列：% of Total Values
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # 根据百分比对表格进行降序排列
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # 打印总结信息：总的列数，有数据缺失的列数
    print("Your selected dataframe has " + str(data.shape[1]) + " columns.\n"
                                                                "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # 返回带有缺失值信息的dataframe
    return mis_val_table_ren_columns


#data3['LVPW'] = data3['LVPW'].fillna(data3['LVPW'].median())
#data3['IVS'] = data3['IVS'].fillna(data3['IVS'].median())
#data3['LogNT'] = data3['LogNT'].fillna(data3['LogNT'].median())
data3['RV'] = data3['RV'].fillna(data3['RV'].median())
data3['EF'] = data3['EF'].fillna(data3['EF'].median())
data1['LVPW'] = data1['LVPW'].fillna(data1['LVPW'].median())
data1['IVS'] = data1['IVS'].fillna(data1['IVS'].median())
data1['LogNT'] = data1['LogNT'].fillna(data1['LogNT'].median())
data1['RV'] = data1['RV'].fillna(data1['RV'].median())
data1['EF'] = data1['EF'].fillna(data1['EF'].median())
data2['LVPW'] = data2['LVPW'].fillna(data2['LVPW'].median())
data2['IVS'] = data2['IVS'].fillna(data2['IVS'].median())
data2['LogNT'] = data2['LogNT'].fillna(data2['LogNT'].median())
data2['RV'] = data2['RV'].fillna(data2['RV'].median())
data2['EF'] = data2['EF'].fillna(data2['EF'].median())

X = data1.iloc[:,3:]
y = data1.iloc[:,2]
Xval = data2.iloc[:,3:]
Yval= data2.iloc[:,2]
Xtrain, Xtest, Ytrain, Ytest  = train_test_split(X, y, test_size=0.2,random_state=1)
oversampler= smote_variants.SMOTE(random_state=1)
X_samp, Y_samp= oversampler.sample(Xtrain.values, Ytrain.values)

X_samp=pd.DataFrame(X_samp)
X_samp.columns=['Gender','NYHA','Occupation','healthinsurance','smoking','drinking','hypertension','familyhistory'
                             ,'allergy','Diseasetime','PCI','arrhythmia','VHD','HLP','DM','centralnervous','COPD','renalinsufficiency'
                             ,'pulsepressure','AGES','BMI','HR','WBC','RBC','RDW','HGB','PLT','NEU','N','ALT','AST','TBIL','DBIL','IBIL'
                            ,'ALP','@GT','TBA','GLU','TC','TG','HDLC','LDLC','BUN','UA','K','NA','CL','CYSC','FT3','FT4','TSH','TNI','HBA1C'
                             ,'LogNT','SG','AF','LA','IVS','LVDD','LVPW','RV','EF','DDF','PE','WMA','MR','TR','PVS1AI','PVI','MS','TS','AS'
                             ,'PVS','PHC','ascites','pleuraleffusion','INFECTION','EDEMA','ECG','BPM','oxygen','antiplatelet','statins'
                             ,'Nitrates','CAantagonist','Betablockers','ACEIARB','Aldosterone','diuretic','Psychiatric']
X_ensamp=X_samp[['AGES', 'AST','BMI', 'Diseasetime', 'EF','FT3', 'FT4','GLU', 'HBA1C','LVDD','MR','RV','TBIL',  'TG', 'TNI', 'TSH','UA', 'WMA','hypertension', 'pulsepressure']]
Xtest=Xtest[['AGES', 'AST','BMI', 'Diseasetime', 'EF','FT3', 'FT4','GLU', 'HBA1C','LVDD','MR','RV','TBIL',  'TG', 'TNI', 'TSH','UA', 'WMA','hypertension', 'pulsepressure']]
Xval=Xval[['AGES', 'AST','BMI', 'Diseasetime', 'EF','FT3', 'FT4','GLU', 'HBA1C','LVDD','MR','RV','TBIL',  'TG', 'TNI', 'TSH','UA', 'WMA','hypertension', 'pulsepressure']]
xgb =  XGBClassifier(learning_rate=0.27,n_estimators=300)
xgb_ypred=xgb.fit(X_ensamp, Y_samp)
explainer_xgb = shap.TreeExplainer(xgb_ypred)
shap_values_xgb = explainer_xgb.shap_values(X_ensamp)

c=[[AGES, AST,BMI, Diseasetime, EF,FT3, FT4,GLU, HBA1C,LVDD,MR_num,RV,TBIL,  TG, TNI, TSH,UA, WMA_num,hypertension_num, pulsepressure],]
c=np.array(c)
clf3 = XGBClassifier(learning_rate=0.27,n_estimators=300).fit(X_ensamp.values, Y_samp)
a=clf3.predict(c)
a1=clf3.predict_proba(c)
a2=a1[0,a]
a3=a2*100
a4=float('%.2f' % a3)

if a==2:
    a="coronary heart disease";
elif a==1:
    a="Coronary atherosclerosis";
else:
    a = "No coronary stenosis"

X_ensamp.columns=['Ages', 'AST', 'BMI', 'Disease time', 'EF', 'FT3', 'FT4', 'Glucose', 'HBA1C',
       'LVDD', 'MR', 'RV', 'TBIL', 'TG', 'Troponin-I', 'Pulse pressure', 'Uric acid', 'WMA',
       'Hypertension', 'TSH']
features_display=X_ensamp.columns
xgb_ypred=clf3.predict(c)
explainer_xgb = shap.TreeExplainer(clf3)
shap_values_xgb1= explainer_xgb.shap_values(c)
shap.decision_plot(explainer_xgb.expected_value[2], shap_values_xgb1[2],features_display)
array1=np.vstack((shap_values_xgb[2][0:20],shap_values_xgb1[2]))
shap.decision_plot(explainer_xgb.expected_value[2], array1,features_display, highlight=20,show=False)
plt.savefig("P1.png")
st.subheader(f"Predict：{a}")
st.subheader(f"概率2：{a4}%")

image = Image.open('p1.png')
st.image(image, use_column_width=True)