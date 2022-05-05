# coding=utf-8
import joblib
import pandas as pd
import numpy as np
import shap
from PIL import Image
import warnings
import streamlit as st
import matplotlib.pyplot as plt
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


xgb= joblib.load('xgb.model')

features_display=['Ages', 'AST', 'BMI', 'Disease time', 'EF', 'FT3', 'FT4', 'Glucose', 'HBA1C',
       'LVDD', 'MR', 'RV', 'TBIL', 'TG', 'Troponin-I', 'TSH', 'Uric acid', 'WMA',
       'Hypertension','Pulse pressure']


c=[[AGES, AST,BMI, Diseasetime, EF,FT3, FT4,GLU, HBA1C,LVDD,MR_num,RV,TBIL,  TG, TNI, TSH,UA, WMA_num,hypertension_num, pulsepressure],]
c=np.array(c)

xgb_ypred=xgb.predict(c)
explainer_xgb = shap.TreeExplainer(xgb)
shap_values_xgb1= explainer_xgb.shap_values(c)

a=xgb.predict(c)
a=int(a)
a1=xgb.predict_proba(c)
a2=a1[0,a]
a3=a2*100
a4=float('%.2f' % a3)

shap_values_xgb= np.load('test00000001.npy')
array1=np.vstack((shap_values_xgb[a][0:20],shap_values_xgb1[a]))

shap.decision_plot(explainer_xgb.expected_value[a],array1,features_display,highlight=20,show=False)
plt.savefig("licancan0729/chd/main/P1.png",dpi=300, bbox_inches='tight')
if a==2:
    a="coronary heart disease";
elif a==1:
    a="Coronary atherosclerosis";
else:
    a = "No coronary stenosis"

st.subheader(f"Predict：{a}")
st.subheader(f"概率2：{a4}%")

image = Image.open('licancan0729/chd/main/p1.png')
st.image(image)
