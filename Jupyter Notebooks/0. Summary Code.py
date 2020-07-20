#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#######################################################################################
# Project "Developing predictive models for COVID-19 diagnosis in paediatric patients: 
#          A case study about the potentials of Machine Learning in Public Health"
#          By Anna Mas-Casadesús (https://github.com/amascasadesus)
#          July 2020
#######################################################################################


### Import basic modules and dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import grid
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.graph_objs as go 
import plotly.offline as py
import plotly.tools as tls
import warnings
from warnings import simplefilter
warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

symp_raw = pd.read_excel('symptoms_paediatric_original.xlsx')
symp = pd.read_excel('symptoms_paediatric_original.xlsx')
symp_raw_copy = symp.copy() # create checkpoints
symp_copy = symp.copy()


###  DATA PRE-PROCESSING I

# Drop redundant, duplicated, and unuseful variables
symp.drop(['underlying','cxrpn','resp1','o21','linfo1','crp1','neutro1','admission1','sex1'], axis=1, inplace=True) 

# Create a new variable that codes whether the patient has had a known contact with a Covid+ or not
symp['covid_contact'] = symp['diagnosis'].str.contains(pat ='covid+')
symp['covid_contact'] = np.multiply(symp['covid_contact'], 1) # convert boolean list to integer

# Correct `diagnosis` variable misspellings and erase "contact with covid+"" (as info already collected in `covid_contact`)
replace_map_diagnosis = {'diagnosis': {'Covid19':'Covid19', 'Fever without focus':'Fever without focus', 
                                       'Febrile neutropenia':'Febrile neutropenia', 'Common cold':'Common cold',
                                       'Bronchopneumonia':'Bronchopneumonia','Acute pyelonephritis':'Acute pyelonephritis',
                                       'Sepsis':'Sepsis','Common cold and covid+ contact':'Common cold', 'Flu':'Flu',
                                       'Nasal hematoma and OMA':'Nasal hematoma and OMA',
                                       'Non-especified vomiting':'Non-especified vomiting',
                                       'Laryngotracheonbronchitis':'Laryngotracheonbronchitis', 
                                       'Diabetic acidosis':'Diabetic acidosis',
                                       'respiratory insuficiency, contact COVID+':'Respiratory insufficiency',
                                       'Pyoderma gangrenosum':'Pyoderma gangrenosum', 
                                       'Septic shock (intestinal origin)':'Septic shock (intestinal origin)',
                                       'Hepatic transplant':'Hepatic transplant',
                                       'Urinary tract infection':'Urinary tract infection','Sacroiliitis':'Sacroiliitis',
                                       'Bronchiolitis obliterans':'Bronchiolitis obliterans',
                                       'central venous catheter infecciton':'Central venous catheter infection',
                                       'Otitis media':'Otitis media','Bronchiolitis and covid+ contact':'Bronchiolitis',
                                       'Intestinal hemorrhage':'Intestinal haemorrhage', 'Bronchiolitis':'Bronchiolitis',
                                       'Acute respiratory insuficiency':'Acute respiratory insufficiency',
                                       'Aspiration neumonitis':'Aspiration pneumonitis','Miocarditis':'Miocarditis'}}
symp.replace(replace_map_diagnosis, inplace=True)

# Encode the different diagnoses (by order of frequency)
symp['diagnosis_coded'] = symp['diagnosis']
replace_map_diagnosis_coded = {'diagnosis_coded': {'Covid19':1,'Fever without focus':2,'Febrile neutropenia':3, 'Common cold':4,
                                                   'Pneumonia':5,'Bronchopneumonia':6,'Acute pyelonephritis':7,'Sepsis':8,
                                                   'Flu':9,'Bronchiolitis':10,'Laryngotracheobronchitis':11,
                                                   'Central venous catheter infection':12,'Respiratory insufficiency':13,
                                                   'Sacroiliitis':14,'Pyoderma gangrenosum':15,'Hepatic transplant':16,
                                                   'Nasal hematoma and OMA':17,'Non-especified vomiting':18,
                                                   'Aspiration pneumonitis':19,'Otitis media':20,'Miocarditis':21,
                                                   'Diabetic acidosis':22,'Bronchiolitis obliterans':23,
                                                   'Intestinal haemorrhage':24,'Acute respiratory insufficiency':25,
                                                   'Septic shock (intestinal origin)':26,'Urinary tract infection':27}}
symp.replace(replace_map_diagnosis_coded, inplace=True)

# Correct `causal_agent`variable misspellings
replace_map_causal_agent = {'causal_agent': {'SARS-cov2':'SARS-cov2','Rhinovirus':'Rhinovirus', 
                                             'Escherichia coli':'Escherichia coli','Influenza A':'Influenza A',
                                             'Adenovirus':'Adenovirus','S agalactiae':'S agalactiae',
                                             'Influenza B':'Influenza B','E coli':'Escherichia coli',
                                             'metapneumovirus':'Metapneumovirus',
                                             'Klebsiella pneumoniae':'Klebsiella pneumoniae carbapenemase',
                                             'Staphylococcus aureus meticiline resistant':
                                             'Methicillin-resistant Staphylococcus aureus',
                                             'Pseudomonas aeruginosa':'Pseudomonas aeruginosa'}}
symp.replace(replace_map_causal_agent, inplace=True)
symp.causal_agent.value_counts()

# Encode the different causal agents (by order of frequency)
symp['causal_agent_coded'] = symp['causal_agent']
replace_map_causal_agent_coded = {'causal_agent_coded': {'SARS-cov2':1, 'Escherichia coli':2,'Influenza A':3,'Rhinovirus':4,
                                                         'S agalactiae':5,'Methicillin-resistant Staphylococcus aureus':6,
                                                         'Adenovirus':7,'Metapneumovirus':8,'Pseudomonas aeruginosa':9,
                                                         'Klebsiella pneumoniae carbapenemase':10,'Influenza B':11}}
symp.replace(replace_map_causal_agent_coded, inplace=True)

# Calculate diagnosis delay (i.e. time between symptoms onset and diagnosis)
symp['symptomsonset'].replace('asymptomatic', np.nan, inplace=True) # replace asymptomatic values for nan
symp['symptomsonset'] = pd.to_datetime(symp['symptomsonset']) # convert `symptomsonset`to datetime64
diagnosis_delay_td = symp['diagnosisdate'] - symp['symptomsonset'] # td = timedelta
import datetime
diagnosis_delay = diagnosis_delay_td.dt.days # transform datetime to numeric
symp['diagnosis_delay'] = diagnosis_delay # attach the new variable to the dataframe
symp.drop(['symptomsonset','diagnosisdate'], axis=1, inplace=True) # remove `symptomsonset` and `diagnosisdate` as we are 
                                                                        # only interested in `diagnosis_delay`

# Replace `admbyother` variable "." values for 0 = no
symp['admbyother'].replace({'.': 0},inplace=True)

# Replace `cxrpneumonia`variable "." values for 0 = no
symp['cxrpneumonia'].replace({'.': 0},inplace=True)

# Correct `virusname` misspellings
#replace_map_virusname = {'virusname': {'ND':'no','Rhinovirus':'Rhinovirus','Influenza A':'Influenza A',
#                                       'Adenovirus':'Adenovirus','No':'no', 'Adenovirus (a sang)':'Adenovirus',
#                                       'Metapneumovirus':'Metapneumovirus'}}
#symp.replace(replace_map_virusname, inplace=True)

# Create a `other_virus_summary` variable automatically retrieving the presence of viruses from `causal_agent`
# Virus = (SARS-cov2), Rhinovirus (1), Influenza A (2), Metapneumovirus (3), Influenza B (4), Adenovirus (5), no (6)
symp.drop(['virusname'], axis=1, inplace=True) # erase `virusname`
def other_virus(v):
    if (v['causal_agent'] == "Rhinovirus"):
        return 1
    elif (v['causal_agent'] == "Influenza A"):  
        return 2
    elif (v['causal_agent'] == "Metapneumovirus"):
        return 3
    elif (v['causal_agent'] == "Influenza B"):
        return 4
    elif (v['causal_agent'] == "Adenovirus"):
        return 5
    else:
        return 6               
symp['other_virus_summary'] = symp.apply(other_virus, axis=1)

# Create a variable with the labels of `other_virus_summary`
symp['other_virus_summary_labels'] = symp['other_virus_summary']
replace_map_other_virus_summary_labels = {'other_virus_summary_labels': {1:'Rhinovirus',2:'Influenza A',3:'Metapneumovirus',
                                                                         4:'Influenza B',5:'Adenovirus',6:'No'}}
symp.replace(replace_map_other_virus_summary_labels, inplace=True)

# Encode the different categories for `other_virus_summary` into independent yes/no variables via one-hot encoding
symp['other_virus_summary_labels2'] = symp['other_virus_summary_labels'] # duplicate `bacteria_summary_labels' 
symp = pd.get_dummies(symp, columns=['other_virus_summary_labels2'], prefix = ['other_virus']) # apply one-hot enconding
symp.drop(['other_virus_No'], axis=1, inplace=True) # remove `other_virus_no` as it is a duplicate of `othervirus` 

# Create a `bacteria_summary` variable automatically retrieving the presence of bacteria from `causal_agent`
# Bacteria = Escherichia coli (1), Klebsiella pneumoniae carbapenemase (2), S agalactiae (3), Pseudomonas aeruginosa (4),
    # Methicillin-resistant Staphylococcus aureus (5)
def bacteria(v):
    if (v['causal_agent'] == "Escherichia coli"):
        return 1
    elif (v['causal_agent'] == "Klebsiella pneumoniae carbapenemase"):  
        return 2
    elif (v['causal_agent'] == "S agalactiae"):
        return 3
    elif (v['causal_agent'] == "Pseudomonas aeruginosa"):
        return 4
    elif (v['causal_agent'] == "Methicillin-resistant Staphylococcus aureus"):
        return 5
    else:
        return 6               
symp['bacteria_summary'] = symp.apply(bacteria, axis=1)

# Create a variable with the labels of `bacteria_summary`
symp['bacteria_summary_labels'] = symp['bacteria_summary']
replace_map_bacteria_summary_labels = {'bacteria_summary_labels': {1:'Escherichia coli',2:'Klebsiella pneumoniae carbapenemase',
                                                                   3:'S agalactiae',4:'Pseudomonas aeruginosa',
                                                                   5:'Methicillin-resistant Staphylococcus aureus',6:'No'}}
symp.replace(replace_map_bacteria_summary_labels, inplace=True)

# Encode the different categories for `bacteria_summary` via one-hot encoding
symp['bacteria_summary_labels2'] = symp['bacteria_summary_labels'] # duplicate `bacteria_summary_labels' 
symp = pd.get_dummies(symp, columns=['bacteria_summary_labels2'], prefix = ['bacteria']) # apply one-hot enconding
replace_map_bacteria_no = {'bacteria_No': {0:1,1:0}} # values for `bacteria_no` are reversed; correct them
symp.replace(replace_map_bacteria_no, inplace=True)

# Replace `ab` variable missing values for 0 = no
symp['ab'].fillna(value=0, inplace=True)

# Remove `abtype` (26 combinations of unidentified types of antibiotics with no use)
symp.drop(['abtype'], axis=1, inplace=True) 

# Update `age1` (0 <= 1yo, 1 <= 5yo, 2 <= 10yo, 3 > 10yo) with `age` info
def age1(v):
    if (v['age'] <= 1):
        return 0
    elif (v['age'] <= 5):  
        return 1
    elif (v['age'] <= 10):
        return 2
    elif (v['age'] > 10 ):
        return 3
symp['age1'] = symp.apply(age1, axis=1)

# Update `age2` (0 <= 5yo, 1 > 5yo) with `age`info
def age2(v):
    if (v['age'] <= 5):
        return 0
    elif (v['age'] > 5 ):
        return 1
symp['age2'] = symp.apply(age2, axis=1)

# Replace `neutro` variable "0" values for nan
replace_map_neutro = {'neutro': {0:'.'}}
symp.replace(replace_map_neutro, inplace=True)
symp['neutro'].replace('.', np.nan, inplace=True)

# Replace `linfo` variable "0" values for nan
replace_map_linfo = {'linfo': {0:'.'}}
symp.replace(replace_map_linfo, inplace=True)
symp['linfo'].replace('.', np.nan, inplace=True)

# Replace `crp` variable "." values for nan
symp['crp'].replace('.', np.nan, inplace=True)
symp['crp'] = symp.crp.astype(float) # convert to float

# Change names of variables for ease of use purposes
symp.columns = ['id', 'age', 'gender', 'diagnosis_covid', 'diagnosis_summary_labels', 'causal_agent_summary_labels',
       'underlying_conditions', 'immunosupressed', 'pid_sid', 'admission', 'admmission_by_covid',
       'admission_by_other', 'admission_picu', 'respiratory', 'gastrointestinal', 'fever', 'cxr', 'pneumonia',
       'oxigen', 'imv', 'inotropics', 'other_virus', 'antibiotics', 'corticoids', 'death', 'age_group1', 
       'age_group2', 'neutrocytes', 'lymphocytes', 'crp', 'covid_contact', 'diagnosis_summary', 
       'causal_agent_summary', 'diagnosis_delay', 
       'other_virus_summary', 'other_virus_summary_labels', 
       'other_virus_adv', 'other_virus_iva', 
       'other_virus_ivb','other_virus_mpv', 
       'other_virus_rv', 'bacteria_summary', 'bacteria_summary_labels',
       'bacteria_ecoli',
       'bacteria_kpn',
       'bacteria_mrsa', 'bacteria',
       'bacteria_paea', 'bacteria_sagl']

# Standardise continuous variables
symp['age_std'] = symp['age'] # Duplicate the variables to standardise (we want to keep the original values for 
                                    # interpretation & visualisation purposes)
symp['neutrocytes_std'] = symp['neutrocytes']
symp['lymphocytes_std'] = symp['lymphocytes']
symp['crp_std'] = symp['crp'] 
symp['diagnosis_delay_std'] = symp['diagnosis_delay'] 
import sklearn
from sklearn.preprocessing import StandardScaler
variables_to_standardise = ['age_std','neutrocytes_std','lymphocytes_std','crp_std','diagnosis_delay_std'] 
scaler = StandardScaler()
symp[variables_to_standardise] = scaler.fit_transform(symp[variables_to_standardise])

# Fill missing values with MICE method
nan = ['neutrocytes_std','lymphocytes_std','crp_std','diagnosis_delay_std'] # Select variables with nan
symp_nan = symp[nan]
from impyute.imputation.cs import mice
nan_mice = mice(symp_nan.values) # apply MICE
nan_mice.columns = ['neutrocytes_mice_std','lymphocytes_mice_std','crp_mice_std','diagnosis_delay_mice_std'] # rename 
nan_nonstd = ['neutrocytes','lymphocytes','crp','diagnosis_delay'] # repeat same process for variables without rescaling
                                                                        # for interpreation & visualisation purposes
symp_nan_nonstd = symp[nan_nonstd]
nan_nonstd_mice = mice(symp_nan_nonstd.values) 
nan_nonstd_mice = pd.DataFrame(data=nan_nonstd_mice[:,:])
nan_nonstd_mice.columns = ['neutrocytes_mice','lymphocytes_mice','crp_mice','diagnosis_delay_mice']
symp = pd.concat([symp, nan_mice, nan_nonstd_mice], axis=1, sort=False) # mrge the different variables with the main dataset


### DATA PRE-PROCESSING II

# Feature selection I: Remove variables with unuseful, unformative, and duplicated info 
    # and those in conflict with our target (e.g. 'admission_by_covid')
sel_variables = ['gender', 'diagnosis_covid', # select the variables to keep
       'underlying_conditions',
       'immunosupressed', 'pid_sid', 'admission_picu', 'respiratory',
       'gastrointestinal', 'fever', 'cxr', 'pneumonia', 'oxigen', 'imv',
       'inotropics', 'admission', 'other_virus','antibiotics','bacteria',
       'corticoids', 'death', 
       'covid_contact', 
       'other_virus_adv', 'other_virus_iva',
       'other_virus_ivb', 'other_virus_mpv', 'other_virus_rv',
       'bacteria_ecoli',
       'bacteria_kpn', 'bacteria_mrsa', 'bacteria_paea',
       'bacteria_sagl', 'age_std',
       'neutrocytes_mice_std', 'lymphocytes_mice_std', 'crp_mice_std',
       'diagnosis_delay_mice_std']                                                     
symp_sel = symp[sel_variables]

# Feature selection II: Remove variables with less than 20% of the cases for one of the classes
sel_variables = ['gender', 'diagnosis_covid', # select variables to keep
       'underlying_conditions',
       'immunosupressed', 'pid_sid', 'admission_picu', 'respiratory',
       'gastrointestinal', 'fever', 'cxr', 'pneumonia', 'oxigen', 
       'antibiotics', 'corticoids', 'age_std',
       'neutrocytes_mice_std', 'lymphocytes_mice_std', 'crp_mice_std',
       'diagnosis_delay_mice_std']                                                     
symp_sel = symp[sel_variables]

# Create another dataset with the select variables with the non-standardised data for interpretation & visualisation purposes
sel_variables_ns = ['gender', 'diagnosis_covid', 
       'underlying_conditions',
       'immunosupressed', 'pid_sid', 'admission_picu', 'respiratory',
       'gastrointestinal', 'fever', 'cxr', 'pneumonia', 'oxigen', 
       'antibiotics', 'corticoids', 'age',
       'neutrocytes_mice', 'lymphocytes_mice', 'crp_mice',
       'diagnosis_delay_mice']                                                     
symp_sel_ns = symp[sel_variables_ns]

# Define X and y and do the Train/Test split
def ttsplit(df,target):
    """
    Splits X and y into train and test subsets
    df = dataframe
    target = target variable
    """
    from sklearn.model_selection import train_test_split
    X, y = df.drop([target],axis=1), df[target] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=df[target])
                                                     # random_state=42 because we want the same splitting each time run it
                                                     # stratified split = keeps the same proportions of class target for
                                                         # train and test subsets (since target is imbalanced)
    return X_train, X_test, y_train, y_test, X, y
Xsel_train, Xsel_test, ysel_train, ysel_test, Xsel, ysel = ttsplit(symp_sel,'diagnosis_covid') # standardised data
Xselns_train, Xselns_test, yselns_train, yselns_test, Xselns, yselns = ttsplit(symp_sel_ns,'diagnosis_covid') # non-standardised

# Oversample the minority class with the SMOTE-NC method: Train (standardised data)
from imblearn.over_sampling import SMOTENC
from collections import Counter
sm = SMOTENC(random_state=42, 
             categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11,12])
Xsel_trainres, ysel_trainres = sm.fit_resample(Xsel_train,ysel_train)
print('Original dataset (y_train) samples per class {}'.format(Counter(ysel_train)))
print('Resampled dataset (y_trainres) samples per class {}'.format(Counter(ysel_trainres)))

# Oversample the minority class with the SMOTE-NC method: Train (non-standardised data)
sm = SMOTENC(random_state=42, 
             categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11,12])
Xselns_trainres, yselns_trainres = sm.fit_resample(Xselns_train,yselns_train)
print('Original dataset (y_train) samples per class {}'.format(Counter(yselns_train)))
print('Resampled dataset (y_trainres) samples per class {}'.format(Counter(yselns_trainres)))
Xselns_trainres.tail(10)

# Oversample the minority class with the SMOTE-NC method: Test (standardised data)
sm = SMOTENC(random_state=42, 
             categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11,12])
Xsel_testres, ysel_testres = sm.fit_resample(Xsel_test,ysel_test)
print('Original dataset (y_test) samples per class {}'.format(Counter(ysel_test)))
print('Resampled dataset (y_testres) samples per class {}'.format(Counter(ysel_testres)))

# Oversample the minority class with the SMOTE-NC method: Test (non-standardised data)
sm = SMOTENC(random_state=42, 
             categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11,12])
Xselns_testres, yselns_testres = sm.fit_resample(Xselns_test,yselns_test)
print('Original dataset (y_test) samples per class {}'.format(Counter(yselns_test)))
print('Resampled dataset (y_testres) samples per class {}'.format(Counter(yselns_testres)))


### EXPLORATORY DATA ANALYSES

# Analysis of variables with respect to frequency and Covid status: Raw dataset
symp_raw_copy2 = symp_raw.copy()
symp_raw_copy2.drop(['covi19'], axis=1, inplace=True) # drop target variable
var = symp_raw_copy2.columns # get features names
var = list(var)
freqVar = symp_raw.shape[0] - symp_raw[var].isnull().sum() # create Covid+/- frequencies
covPos = (symp_raw[symp_raw['covi19']==1].shape[0]
         -symp_raw[symp_raw['covi19']==1][var].isnull().sum())/(symp_raw.shape[0] - symp_raw[var].isnull().sum())
covNeg = (symp_raw[symp_raw['covi19']==0].shape[0]
         -symp_raw[symp_raw['covi19']==0][var].isnull().sum())/(symp_raw.shape[0] - symp_raw[var].isnull().sum())
fig, axs = plt.subplots(2, 1, figsize=(25,8))
fig.patch.set_facecolor('white')
pFreqVar = axs[0].bar(var, freqVar, color='grey', )  # frequency plot
axs[0].set_title("(a) Frequency of variables available")
axs[0].get_xaxis().set_ticks([])
pCovPos = axs[1].bar(var, covPos, color='#1e90ff') # percentage of Covid cases over total tests performed
pCovNeg = axs[1].bar(var, covNeg, bottom=covPos, color='#ff8c00')
plt.xticks(var, var, rotation='vertical')
axs[1].set_title("(b) Percentage of Covid cases over total variables available")
axs[1].legend(["CovidPos", "CovidNeg"])
plt.xticks(var, var, rotation='vertical')
plt.subplots_adjust(hspace=0.2) 
fig.suptitle("Analysis of variables with respect to frequency and Covid status - Raw dataset")
plt.plot()

# Analysis of variables with respect to frequency and Covid status: Pre-processed dataset I
symp_copy2 = symp.copy()
symp_copy2.drop(['diagnosis_covid'], axis=1, inplace=True) 
var = symp_copy2.columns
var = list(var)
freqVar = symp.shape[0] - symp[var].isnull().sum()
covPos = (symp[symp['diagnosis_covid']==1].shape[0]
         -symp[symp['diagnosis_covid']==1][var].isnull().sum())/(symp.shape[0] - symp[var].isnull().sum())
covNeg = (symp[symp['diagnosis_covid']==0].shape[0]
         -symp[symp['diagnosis_covid']==0][var].isnull().sum())/(symp.shape[0] - symp[var].isnull().sum())
fig, axs = plt.subplots(2, 1, figsize=(25,8))
fig.patch.set_facecolor('white')
pFreqVar = axs[0].bar(var, freqVar, color='grey', )  
axs[0].set_title("(a) Frequency of variables available")
axs[0].get_xaxis().set_ticks([])
pCovPos = axs[1].bar(var, covPos, color='#1e90ff')
pCovNeg = axs[1].bar(var, covNeg, bottom=covPos, color='#ff8c00')
plt.xticks(var, var, rotation='vertical')
axs[1].set_title("(b) Percentage of Covid cases over total variables available")
axs[1].legend(["CovidPos", "CovidNeg"])
plt.xticks(var, var, rotation='vertical')
plt.subplots_adjust(hspace=0.2) 
fig.suptitle("Analysis of variables with respect to frequency and Covid status - Pre-processed I dataset")
plt.plot()

# Analysis of variables with respect to frequency and Covid status: Pre-processed dataset II
symp_sel_copy2 = symp_sel.copy()
symp_sel_copy2.drop(['diagnosis_covid'], axis=1, inplace=True) 
var = symp_sel_copy2.columns
var = list(var)
freqVar = symp_sel.shape[0] - symp_sel[var].isnull().sum()
covPos = (symp_sel[symp_sel['diagnosis_covid']==1].shape[0]
         -symp_sel[symp_sel['diagnosis_covid']==1][var].isnull().sum())/(symp_sel.shape[0] - symp_sel[var].isnull().sum())
covNeg = (symp_sel[symp_sel['diagnosis_covid']==0].shape[0]
         -symp_sel[symp_sel['diagnosis_covid']==0][var].isnull().sum())/(symp_sel.shape[0] - symp_sel[var].isnull().sum())
fig, axs = plt.subplots(2, 1, figsize=(25,8))
fig.patch.set_facecolor('white')
pFreqVar = axs[0].bar(var, freqVar, color='grey', )  
axs[0].set_title("(a) Frequency of variables available")
axs[0].get_xaxis().set_ticks([])
pCovPos = axs[1].bar(var, covPos, color='#1e90ff')
pCovNeg = axs[1].bar(var, covNeg, bottom=covPos, color='#ff8c00')
plt.xticks(var, var, rotation='vertical')
axs[1].set_title("(b) Percentage of Covid cases over total variables available")
axs[1].legend(["CovidPos", "CovidNeg"])
plt.xticks(var, var, rotation='vertical')
plt.subplots_adjust(hspace=0.2) 
fig.suptitle("Analysis of variables with respect to frequency and Covid status")
plt.plot()

# Categorical variables plot
symp_sel_copy3 = symp_sel.copy()
symp_sel_copy3.drop(['diagnosis_covid','age_std','neutrocytes_mice_std','lymphocytes_mice_std', # drop numerical variables
                    'crp_mice_std','diagnosis_delay_mice_std'], axis=1, inplace=True) 
cat_var = symp_sel_copy3.columns # get features names
cat_var = list(cat_var)
cat_var[0] = "male" # rename `gender`
symp_sel.columns = ['male', 'diagnosis_covid', 'underlying_conditions', 'immunosupressed',
       'pid_sid', 'admission_picu', 'respiratory', 'gastrointestinal', 'fever',
       'cxr', 'pneumonia', 'oxigen', 'antibiotics', 'corticoids', 'age_std',
       'neutrocytes_mice_std', 'lymphocytes_mice_std', 'crp_mice_std',
       'diagnosis_delay_mice_std']
labels = ["CovidNeg", "CovidPos"] # plot 
newPal = ["#ff8c00", "#1e90ff"]
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15,30))
r = 0 # Index row
c = 0 # Index col
cat_var 
for f in cat_var:
    # Count plot
    sns.countplot(x=f, hue='diagnosis_covid', data=symp_sel,ax=axes[r][c], palette=newPal)
    # Plot configuration
    axes[r][c].legend(labels, loc='upper right')
    axes[r][c].set_xticklabels(["No", "Yes"])
    # Index control
    c += 1
    if c > 1:
        c = 0
        r += 1

# Numerical variables plot
symp_sel_copy4 = symp_sel.copy()
symp_sel_copy4.drop(['male', 'diagnosis_covid', 'underlying_conditions', 'immunosupressed', # drop categorical variables
                     'pid_sid', 'admission_picu', 'respiratory', 'gastrointestinal', 'fever',
                     'cxr', 'pneumonia', 'oxigen', 'antibiotics', 'corticoids'], axis=1, inplace=True) 
num_var = symp_sel_copy4.columns # get features names
num_var = list(num_var)
labels = ["CovidNeg", "CovidPos"] # plot
newPal = ["#ff8c00", "#1e90ff"]
sns.set(style="white", palette=newPal, color_codes=False)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15))
r = 0 # Index row
c = 0 # Index col
for f in num_var:
    # Distribution plot
    sns.distplot(symp_sel[symp_sel['diagnosis_covid'] == 0][f],label="CovidNeg",ax=axes[r][c])
    sns.distplot(symp_sel[symp_sel['diagnosis_covid'] == 1][f],label="CovidPos",ax=axes[r][c])
    # Plot configurations
    axes[r][c].legend(labels, loc='upper right')
    # Index control
    c += 1
    if c > 1:
        c = 0
        r += 1
        

### DATA MODELLING I: PIPELINES

# All models pipeline
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import joblib
import time
import xgboost as xgb
start = time.time()
# Define the combinations of xy variables (i.e. subsets)
xy_sel = [Xsel_trainres, Xsel_test, ysel_trainres, ysel_test]
xy_selor = [Xsel_train, Xsel_test, ysel_train, ysel_test]
xy_list = [xy_sel, xy_selor]
xy_liststr = ["xy_sel", "xy_selor"]
# Built pipelines
pipe_lr = Pipeline([('clf', LogisticRegression())])
pipe_knn = Pipeline([('clf', KNeighborsClassifier())])
pipe_svm = Pipeline([('clf', svm.SVC())])
pipe_dt = Pipeline([('clf', DecisionTreeClassifier())])
pipe_rf = Pipeline([('clf', RandomForestClassifier())])
pipe_xgb = Pipeline([('clf', xgb.XGBClassifier())])
# Set grid search parameters
param_range_1 = [int(x) for x in np.linspace(0.001, 10, num=100)] # cost
param_range_2 = [int(x) for x in np.linspace(2, 10, num=9)] # n_neigghbors, max_dexpth, min_samples_leaf 
param_range_3 = [int(x) for x in np.linspace(2, 20, num=19)] # min_samples_split
param_range_4 = [int(x) for x in np.linspace(0, 1, num=20)] # learning_rate
param_range_5 = [int(x) for x in np.linspace(0.2, 0.8, num=6)] # colsample_bytree
param_range_6 = [int(x) for x in np.linspace(0, 5, num=10)] # min_child_weight
param_range_7 = [int(x) for x in np.linspace(0, 0.1, num=5)] # gamma
param_range_8 = [int(x) for x in np.linspace(3, 10, num=8)] # n_estimators
grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                   'clf__C': param_range_1,
                   'clf__solver': ['liblinear', 'saga', 'lbfgs']}] 
grid_params_knn = [{'clf__n_neighbors': param_range_2}]
grid_params_svm = [{'clf__kernel': ['linear', 'rbf'],
                    'clf__gamma': ['auto','scale'],
                    'clf__C': param_range_1}]
grid_params_dt = [{'clf__criterion': ['gini', 'entropy'],
                   'clf__splitter': ['best','random'],
                   'clf__min_samples_leaf': param_range_2,
                   'clf__max_depth': param_range_2, 
                   'clf__min_samples_split': param_range_3}]
grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],              
                   'clf__min_samples_leaf': param_range_2, 
                   'clf__max_depth': param_range_2, 
                   'clf__min_samples_split': param_range_3}]            
grid_params_xgb = [{'clf__objective': ['reg:logistic'],
                    'clf__subsample': [0.5],
                    'clf__max_depth': param_range_2,
                    'clf__learning_rate': param_range_4,
                    'clf__colsample_bytree': param_range_5,
                    'clf__min_child_weight': param_range_5,
                    'clf__gamma': param_range_7,
                    'clf__n_estimators': param_range_8}] 
# Built grid searches
kfold = StratifiedKFold(n_splits=10, random_state=42)
gs_lr = RandomizedSearchCV(estimator=pipe_lr,
                     param_distributions=grid_params_lr,
                     scoring='f1_micro',
                     cv=kfold) 
gs_knn = RandomizedSearchCV(estimator=pipe_knn,
                     param_distributions=grid_params_knn,
                     scoring='f1_micro',
                     cv=kfold)
gs_svm = RandomizedSearchCV(estimator=pipe_svm,
                      param_distributions=grid_params_svm,
                      scoring='f1_micro',
                      cv=kfold)
gs_dt = RandomizedSearchCV(estimator=pipe_dt,
                     param_distributions=grid_params_dt,
                     scoring='f1_micro',
                     cv=kfold) 
gs_rf = RandomizedSearchCV(estimator=pipe_rf,
                     param_distributions=grid_params_rf,
                     scoring='f1_micro',
                     cv=kfold) 
gs_xgb = RandomizedSearchCV(estimator=pipe_xgb,
                     param_distributions=grid_params_xgb,
                     scoring='f1_micro',
                     cv=kfold) 
# List of pipelines for ease of iteration
grids = [gs_lr, gs_knn, gs_svm, gs_dt, gs_rf, gs_xgb]
# Dictionary of pipelines and classifiers for ease of reference
grid_dict = {0: 'Logistic Regression', 
             1: 'k-Nearest Neighbors', 
             2: 'Support Vector Machine', 
             3: 'Decision Tree', 
             4: 'Random Forest',
             5: 'eXtra Gradient Boosting'}
# Fit the grid search objects
print('Performing model optimisations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
def getxsys(xy):  
    """
    Gets X_train, X_test, y_test, and y_test of each specific subset
    """
    X_train = xy[0]
    X_test = xy[1]
    y_train = xy[2]
    y_test = xy[3]
    return X_train, X_test, y_train, y_test # Outside the for loop to optimise processing
rounds_list = ["1st", "2nd" , "3rd", "4th", "5th"]
rounds_list_iter = iter(rounds_list)  # Iterates over rounds_list
for _ in range(0,5):
    round_num = next(rounds_list_iter) # Takes one round item per iteration in rounds_list_iter, sequentially 
    print('\nRound:', round_num)
    xy_liststr_iter = iter(xy_liststr) # Iteraves over xy_liststr
    for xy in xy_list:
        X_train, X_test, y_train, y_test = getxsys(xy)
        xy_str = next(xy_liststr_iter) # Takes one list name per iteration in xy_liststr_iter, sequentially 
        print('\n', xy_str)
        for idx, gs in enumerate(grids):
            print('\nEstimator: %s' % (grid_dict[idx]), xy_str)
            # Fit grid search
            gs.fit(X_train, y_train)
            # Best params
            print('Best params: %s' % gs.best_params_)
            # Best training data accuracy
            print('Best training F1 score: %.3f' % gs.best_score_)
            # Predict on test data with best params
            y_pred = gs.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set F1 score for best params: %.3f ' % f1_score(y_test, y_pred, average='micro'))
            # Track best (highest test accuracy) model
            if accuracy_score(y_test, y_pred) > best_acc:
                best_acc = accuracy_score(y_test, y_pred)
                best_gs = gs
                best_clf = idx
                best_xy = xy_str
    print('\nClassifier with best test set F1 score: %s' % grid_dict[best_clf], best_xy)
end = time.time()
print('\nTotal execution time:', ((end - start)/60), 'min')

# Reduced models pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
start = time.time()
# Define the combinations of xy variables (i.e. subsets)
xy_sel = [Xsel_trainres, Xsel_test, ysel_trainres, ysel_test]
xy_selor = [Xsel_train, Xsel_test, ysel_train, ysel_test]
xy_list = [xy_sel, xy_selor]
xy_liststr = ["xy_sel", "xy_selor"]
# Built pipelines
pipe_dummy = Pipeline([('clf', DummyClassifier(random_state=42))])
pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=42))])
# Set grid search parameters
param_range_2 = [int(x) for x in np.linspace(2, 10, num=9)] # n_neigghbors, max_dexpth, min_samples_leaf 
param_range_3 = [int(x) for x in np.linspace(2, 20, num=19)] # min_samples_split
grid_params_dummy = [{'clf__strategy': ['most_frequent']}]
grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],              
                   'clf__min_samples_leaf': param_range_2, 
                   'clf__max_depth': param_range_2, 
                   'clf__min_samples_split': param_range_3}]            
# Built grid searches
kfold = StratifiedKFold(n_splits=10, random_state=42)
gs_dummy = GridSearchCV(estimator=pipe_dummy,
                     param_grid=grid_params_dummy,
                     scoring='f1_micro',
                     cv=kfold) 
gs_rf = GridSearchCV(estimator=pipe_rf,
                     param_grid=grid_params_rf,
                     scoring='f1_micro',
                     cv=kfold) 
# List of pipelines for ease of iteration
grids = [gs_dummy, gs_rf]
# Dictionary of pipelines and classifiers for ease of reference
grid_dict = {0: 'Performance baseline', 
             1: 'Random Forest'}
# Fit the grid search objects
print('Performing model optimisations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
def getxsys(xy):  
    """
    Gets X_train, X_test, y_test, and y_test of each specific subset
    """
    X_train = xy[0]
    X_test = xy[1]
    y_train = xy[2]
    y_test = xy[3]
    return X_train, X_test, y_train, y_test # Outside the for loop to optimise processing
xy_liststr_iter = iter(xy_liststr) # Iterates over xy_liststr
for xy in xy_list:
    X_train, X_test, y_train, y_test = getxsys(xy)
    xy_str = next(xy_liststr_iter) # Takes one list name per iteration in xy_liststr_iter, sequentially
    print('\n', xy_str)
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % (grid_dict[idx]), xy_str)
        # Fit grid search
        gs.fit(X_train, y_train)
        # Best params
        print('Best params: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training F1 score: %.3f' % gs.best_score_)
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        # Test data accuracy of model with best params
        print('Test set F1 score for best params: %.3f ' % f1_score(y_test, y_pred, average='micro'))
        # Track best (highest test accuracy) model
        if accuracy_score(y_test, y_pred) > best_acc:
            best_acc = accuracy_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx
            best_xy = xy_str
print('\nClassifier with best test set F1 score: %s' % grid_dict[best_clf], best_xy)
end = time.time()
print('\nTotal models executed: 6156')
print('\nTotal execution time:', ((end - start)/60), 'min')


### DATA MODELLING II: GLOBAL SURROGATE & EVALUATION

# Global Surrogate Method
# Random Forest (RF) models definition
from sklearn.ensemble import RandomForestClassifier
rfmodel_ov = RandomForestClassifier(criterion='gini', # Oversampled data - Oversampled model parameters · OVERSAMPLED model (OV)
                                    max_depth=2,
                                    min_samples_leaf=2,
                                    min_samples_split=2,
                                    random_state=42)
rfmodel_ov.fit(Xsel_trainres, ysel_trainres)
rfmodel_or = RandomForestClassifier(criterion='entropy', # Original data - Original model parameters · ORIGINAL model (OR)
                                    max_depth=3,
                                    min_samples_leaf=3,
                                    min_samples_split=8,
                                    random_state=42)
rfmodel_or.fit(Xsel_train, ysel_train)
# RF predictions
yselrf_trainres = rfmodel_ov.predict(Xsel_trainres) # OV
yselrf_train = rfmodel_or.predict(Xsel_train) # OR
# Decision Tree (DT) models definition
from sklearn.tree import DecisionTreeClassifier # OV
dtmodel_ov = DecisionTreeClassifier(random_state=42)
dtmodel_ov.fit(Xsel_trainres, yselrf_trainres)
dtmodel_or = DecisionTreeClassifier(random_state=42) # OR
dtmodel_or.fit(Xsel_train, yselrf_train)
dtmodelns_ov = DecisionTreeClassifier(random_state=42) # OV with non-standardised data (interpreation & visualisation)
dtmodelns_ov.fit(Xselns_trainres, yselrf_trainres)
dtmodelns_or = DecisionTreeClassifier(random_state=42) # OR with non-standardised data
dtmodelns_or.fit(Xselns_train, yselrf_train)

# Performance metrics
def performance_metrics(model, X_train, y_train, X_test, y_test, train=True, cv=True):
    """
    Evaluates a classification model
    train = True to evaluate train subset, = False test subset
    cv = True performs stratified cross-validation
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score 
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold
    scoring = {'acc': 'accuracy',
               'prec_micro': 'precision_micro',
               'rec_micro': 'recall_micro',
               'f1_micro': 'f1_micro',
               'auc':'roc_auc'}    
    if train==True:
        if cv==True:
            kfold=StratifiedKFold(n_splits=10, random_state=42)
            scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=kfold)
            ypredTrain = model.predict(X_train)
            Acc_train = scores['test_acc'].mean()
            Precision_train = scores['test_prec_micro'].mean()
            Recall_train = scores['test_rec_micro'].mean()
            F1_train = scores['test_f1_micro'].mean()
            AUC_train = scores['test_auc'].mean()
            conf_matrix_train = confusion_matrix(y_train, ypredTrain)
            class_report = classification_report(y_train, ypredTrain)
            print("TRAIN:\n===========================================")
            print(f"CV - Accuracy : {Acc_train:.2f}\n")
            print(f"CV - Precision: {Precision_train:.2f}\n")
            print(f"CV - Recall: {Recall_train:.2f}\n")
            print(f"CV - F1 score: {F1_train:.2f}\n")   
            print(f"CV - AUC score: {AUC_train:.2f}\n") 
            print(f"Confusion Matrix:\n {conf_matrix_train}\n")
            print(f"Classification Report:\n {class_report}\n")           
        elif cv==False:
            scores = cross_validate(model, X_train, y_train, scoring=scoring)
            ypredTrain = model.predict(X_train)
            Acc_train = scores['test_acc'].mean()
            Precision_train = scores['test_prec_micro'].mean()
            Recall_train = scores['test_rec_micro'].mean()
            F1_train = scores['test_f1_micro'].mean()
            AUC_train = scores['test_auc'].mean()
            conf_matrix_train = confusion_matrix(y_train, ypredTrain)
            class_report = classification_report(y_train, ypredTrain)
            print("TRAIN:\n===========================================")
            print(f"CV - Accuracy : {Acc_train:.2f}\n")
            print(f"CV - Precision: {Precision_train:.2f}\n")
            print(f"CV - Recall: {Recall_train:.2f}\n")
            print(f"CV - F1 score: {F1_train:.2f}\n")   
            print(f"CV - AUC score: {AUC_train:.2f}\n")  
            print(f"Confusion Matrix:\n {conf_matrix_train}\n")
            print(f"Classification Report:\n {class_report}\n")
    elif train==False:
        if cv==True:
            kfold=StratifiedKFold(n_splits=10, random_state=42)
            scores = cross_validate(model, X_test, y_test, scoring=scoring, cv=kfold)
            ypredTest = model.predict(X_test)
            Acc_test = scores['test_acc'].mean()
            Precision_test = scores['test_prec_micro'].mean()
            Recall_test = scores['test_rec_micro'].mean()
            F1_test = scores['test_f1_micro'].mean()
            AUC_test = scores['test_auc'].mean()
            conf_matrix_test = confusion_matrix(y_test, ypredTest)
            class_report = classification_report(y_test, ypredTest)        
            print("TEST:\n===========================================")
            print(f"CV - Accuracy : {Acc_test:.2f}\n")
            print(f"CV - Precision: {Precision_test:.2f}\n")
            print(f"CV - Recall: {Recall_test:.2f}\n")
            print(f"CV - F1 score: {F1_test:.2f}\n")    
            print(f"CV - AUC score: {AUC_test:.2f}\n")   
            print(f"Confusion Matrix:\n {conf_matrix_test}\n")
            print(f"Classification Report:\n {class_report}\n")
        elif cv==False:
            scores = cross_validate(model, X_test, y_test, scoring=scoring)
            ypredTest = model.predict(X_test)
            Acc_test = scores['test_acc'].mean()
            Precision_test = scores['test_prec_micro'].mean()
            Recall_test = scores['test_rec_micro'].mean()
            F1_test = scores['test_f1_micro'].mean()
            AUC_test = scores['test_auc'].mean()
            conf_matrix_test = confusion_matrix(y_test, ypredTest)
            class_report = classification_report(y_test, ypredTest)        
            print("TEST:\n===========================================")
            print(f"CV - Accuracy : {Acc_test:.2f}\n")
            print(f"CV - Precision: {Precision_test:.2f}\n")
            print(f"CV - Recall: {Recall_test:.2f}\n")
            print(f"CV - F1 score: {F1_test:.2f}\n")    
            print(f"CV - AUC score: {AUC_test:.2f}\n")   
            print(f"Confusion Matrix:\n {conf_matrix_test}\n")
            print(f"Classification Report:\n {class_report}\n")   
# OV train
print('Oversampled Model\n')
print(performance_metrics(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_test, ysel_test, train=True, cv=True))
# OV test
print('Oversampled Model\n')
print(performance_metrics(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_test, ysel_test, train=False, cv=False))
# OR train
print('Original Model\n')
print(performance_metrics(dtmodel_or, Xsel_train, yselrf_train, Xsel_test, ysel_test, train=True, cv=False))
# OR test
print('Original Model\n')
print(performance_metrics(dtmodel_or, Xsel_train, yselrf_train, Xsel_test, ysel_test, train=False, cv=False))
# Projected (PJ; i.e. oversampled test)
print('Projected Model\n')
print(performance_metrics(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_testres, ysel_testres, train=False, cv=True))

# Performance figures
def pr_curve(model, X_train, y_train, X_test, y_test, train=True):
    """
    Plots a precision/recall curve of a given classification model
    train = True to evaluate train subset, = False test subset
    """
    from sklearn.metrics import precision_recall_curve
    if train==True:
        ypredTrain = model.predict(X_train)   
        precisions, recalls, thresholds = precision_recall_curve(y_train, ypredTrain)
        plt.plot(precisions, recalls, linewidth=3, color='r', linestyle='-')
        plt.rc('xtick', labelsize=10)    
        plt.rc('ytick', labelsize=10)  
        plt.xlabel("Precision", size=12)
        plt.ylabel("Recall", size=12)
        plt.grid()
        plt.rcParams['figure.facecolor'] = '#F2F3F4'            
        plt.rcParams['axes.facecolor'] = '#F2F3F4'                           
        plt.title("PR Curve: Precision/Recall Trade-off\n\n(Train)\n", size=14)          
        plt.show()
    elif train==False:
        ypredTest = model.predict(X_test)
        precisions, recalls, thresholds = precision_recall_curve(y_test, ypredTest)
        plt.plot(precisions, recalls, linewidth=3, color='b', linestyle='-')
        plt.rc('xtick', labelsize=10)    
        plt.rc('ytick', labelsize=10)  
        plt.xlabel("Precision", size=12)
        plt.ylabel("Recall", size=12)
        plt.grid()
        plt.rcParams['figure.facecolor'] = '#F2F3F4'            
        plt.rcParams['axes.facecolor'] = '#F2F3F4'
        plt.title("PR Curve: Precision/Recall Trade-off\n\n(Test)\n", size=14)
        plt.show()
pr_curve(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_test, ysel_test, train=True) # OV train
pr_curve(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_test, ysel_test, train=False) # OV test
pr_curve(dtmodel_or, Xsel_train, yselrf_train, Xsel_test, ysel_test, train=True) # OR train
pr_curve(dtmodel_or, Xsel_train, yselrf_train, Xsel_test, ysel_test, train=False) # OR test
pr_curve(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_testres, ysel_testres, train=False) # PJ test
def roc_curve(model, X_train, y_train, X_test, y_test, train=True):  
    """
    Plots a ROC curve of a given classification model
    train = True to evaluate train subset, = False test subset
    """
    from sklearn.metrics import roc_curve
    if train==True:
        ypredTrain = model.predict(X_train)
        fpr, tpr, thresholds = roc_curve(y_train, ypredTrain)
        plt.plot(fpr, tpr, linewidth=3, label=None, color='r', linestyle='-')
        plt.rc('xtick', labelsize=10)    
        plt.rc('ytick', labelsize=10)  
        plt.xlabel('False Positive Rate', size=12)
        plt.ylabel('True Positive Rate', size=12)
        plt.grid()
        plt.rcParams['figure.facecolor'] = '#F2F3F4'            
        plt.rcParams['axes.facecolor'] = '#F2F3F4'  
        plt.title("ROC Curve: Sensitivity/Specificity Trade-off\n\n(Train)\n", size=14)
        plt.show()
    elif train==False:
        ypredTest = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, ypredTest)
        plt.plot(fpr, tpr, linewidth=3, label=None, color='b', linestyle='-')
        plt.rc('xtick', labelsize=10)    
        plt.rc('ytick', labelsize=10)  
        plt.xlabel('False Positive Rate', size=12)
        plt.ylabel('True Positive Rate', size=12)
        plt.grid()
        plt.rcParams['figure.facecolor'] = '#F2F3F4'            
        plt.rcParams['axes.facecolor'] = '#F2F3F4'
        plt.title('ROC Curve: Sensitivity/Specificity Trade-off\n\n(Test)\n', size=14)
        plt.show()
roc_curve(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_test, ysel_test, train=True) # OV train
roc_curve(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_test, ysel_test, train=False) # OV test
roc_curve(dtmodel_or, Xsel_train, yselrf_train, Xsel_test, ysel_test, train=True) # OR train
roc_curve(dtmodel_or, Xsel_train, yselrf_train, Xsel_test, ysel_test, train=False) # OR test
roc_curve(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_testres, ysel_testres, train=False) # PJ test
def conf_matrix(model, X_train, y_train, X_test, y_test, train=True): 
    """
    Plots a confusion matrix of a given classification model
    train = True to evaluate train subset, = False test subset
    """
    from sklearn.metrics import confusion_matrix
    import itertools
    if train==True:    
        ypredTrain = model.predict(X_train)
        cm = confusion_matrix(y_train, ypredTrain)
        def plot_conf_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Reds):
            plt.figure(figsize = (5, 5))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title, size = 14)
            plt.colorbar(aspect=4)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=0, size = 10)
            plt.yticks(tick_marks, classes, size = 10)
            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), fontsize = 14,
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                plt.grid(b=None)
            plt.tight_layout()
            plt.ylabel('True label', size = 12)
            plt.xlabel('Predicted label', size = 12)
        plot_conf_matrix(cm, classes = ['Covid-', 'Covid+'], 
                                  title = 'Confusion Matrix\n\n(Train)\n')
    elif train==False:
        ypredTest = model.predict(X_test)
        cm = confusion_matrix(y_test, ypredTest)
        def plot_conf_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
            plt.figure(figsize = (5, 5))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title, size = 14)
            plt.colorbar(aspect=4)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=0, size = 10)
            plt.yticks(tick_marks, classes, size = 10)
            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), fontsize = 14,
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                plt.grid(b=None)
            plt.tight_layout()
            plt.ylabel('True label', size = 12)
            plt.xlabel('Predicted label', size = 12)
        plot_conf_matrix(cm, classes = ['Covid-', 'Covid+'], 
                                  title = 'Confusion Matrix\n\n(Test)\n')        
conf_matrix(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_test, ysel_test, train=True) # OV train          
conf_matrix(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_test, ysel_test, train=False) # OV test
conf_matrix(dtmodel_or, Xsel_train, yselrf_train, Xsel_test, ysel_test, train=True) # OR train
conf_matrix(dtmodel_or, Xsel_train, yselrf_train, Xsel_test, ysel_test, train=False) # OR test
conf_matrix(dtmodel_ov, Xsel_trainres, yselrf_trainres, Xsel_testres, ysel_testres, train=False) # PJ test

# Predictions on test
dtmodel_ov_classPred = dtmodel_ov.predict(Xsel_test) # OV model
print('Classification predictions: \n', dtmodel_ov_classPred)
dtmodel_ov_probPred = dtmodel_ov.predict_proba(Xsel_test)[:, 1]
print('Predictions probabilities: \n', dtmodel_ov_probPred)
dtmodel_or_classPred = dtmodel_or.predict(Xsel_test) # OR model
print('Classification predictions: \n', dtmodel_or_classPred)
dtmodel_or_probPred = dtmodel_or.predict_proba(Xsel_test)[:, 1]
print('Predictions probabilities: \n', dtmodel_or_probPred)
dtmodel_ov_classPred = dtmodel_ov.predict(Xsel_testres) # PJ model
print('Classification predictions: \n', dtmodel_ov_classPred)
dtmodel_ov_probPred = dtmodel_ov.predict_proba(Xsel_testres)[:, 1]
print('Predictions probabilities: \n', dtmodel_ov_probPred)


### MODEL INTERPRETATION

# Decision Tree figures
featurenames = ['gender', 'preconditions', 'immunosupressed', 'pidsid', # define features names
       'admpicu', 'respiratory', 'gastrointestinal', 'fever', 'cxr',
       'pneumonia', 'oxigen', 'antibiotics', 'corticoids', 'age',
       'neutrocytes', 'lymphocytes', 'crp',
       'diagnosisdelay']
classnames = ['CovidNeg','CovidPos'] # define class names
from sklearn import tree
import graphviz
# OV standardised
dot_data = tree.export_graphviz(dtmodel_ov, 
                                out_file=None,  
                                feature_names=featurenames,
                                class_names=classnames, 
                                filled=True, 
                                rounded=True, 
                                special_characters=True) 
graph = graphviz.Source(dot_data)  
graph.render('dtree_ov',view=True) # creates and opens pdf file in working folder
# OV non-standardised
dot_data = tree.export_graphviz(dtmodelns_ov, 
                                out_file=None,  
                                feature_names=featurenames,
                                class_names=classnames, 
                                filled=True, 
                                rounded=True, 
                                special_characters=True) 
graph = graphviz.Source(dot_data)  
graph.render('dtree_ovns',view=True)
# OR standardised
dot_data = tree.export_graphviz(dtmodel_or, 
                                out_file=None,  
                                feature_names=featurenames,
                                class_names=classnames, 
                                filled=True, 
                                rounded=True, 
                                special_characters=True) 
graph = graphviz.Source(dot_data)  
graph.render('dtree_or',view=True)
# OR non-standardised
dot_data = tree.export_graphviz(dtmodelns_or, 
                                out_file=None,  
                                feature_names=featurenames,
                                class_names=classnames, 
                                filled=True, 
                                rounded=True, 
                                special_characters=True) 
graph = graphviz.Source(dot_data)  
graph.render('dtree_orns',view=True)

# Feature importance analyses & figures (SHAP method)
#conda install -c conda-forge (if cannot be installed through Anaconda console)
import shap
# OV standardised - Feature Importance Plot: Global Interpretability
shap_values_dtov = shap.TreeExplainer(dtmodel_ov).shap_values(Xsel_trainres)
shap.summary_plot(shap_values_dtov, Xsel_trainres, plot_type="bar")
# OV standardised - Individual SHAP Value Plot — Local Interpretability 
X_output_dtov = Xsel_test.copy()
X_output_dtov.loc[:,'predict'] = np.round(dtmodel_ov.predict(X_output_dtov),2)
explainer_dtov = shap.TreeExplainer(dtmodel_ov)
choosen_instance_dtov0 = X_output_dtov.loc[[0]] # positive case example
shap_values_dtov0 = explainer_dtov.shap_values(choosen_instance_dtov0)
shap.initjs()
shap.force_plot(explainer_dtov.expected_value[1], shap_values_dtov0[1], choosen_instance_dtov0, 
                show=True, matplotlib=True)
choosen_instance_dtov10 = X_output_dtov.loc[[10]] # negative case example
shap_values_dtov10 = explainer_dtov.shap_values(choosen_instance_dtov10)
shap.initjs()
shap.force_plot(explainer_dtov.expected_value[1], shap_values_dtov10[1], choosen_instance_dtov10, 
                show=True, matplotlib=True, figsize=[27,2.5]) 
# OV non-standardised - Feature Importance Plot: Global Interpretability
shap_values_dtovns = shap.TreeExplainer(dtmodelns_ov).shap_values(Xselns_trainres)
shap.summary_plot(shap_values_dtovns, Xselns_trainres, plot_type="bar")
# OV non- standardised - Individual SHAP Value Plot — Local Interpretability 
X_output_dtovns = Xselns_test.copy()
X_output_dtovns.loc[:,'predict'] = np.round(dtmodelns_ov.predict(X_output_dtovns),2)
explainer_dtovns = shap.TreeExplainer(dtmodelns_ov)
choosen_instance_dtovns0 = X_output_dtovns.loc[[0]] # positive case example
shap_values_dtovns0 = explainer_dtovns.shap_values(choosen_instance_dtovns0)
shap.initjs()
shap.force_plot(explainer_dtovns.expected_value[1], shap_values_dtovns0[1], choosen_instance_dtovns0, 
                show=True, matplotlib=True) 
choosen_instance_dtovns11 = X_output_dtovns.loc[[11]] # negative case example
shap_values_dtovns11 = explainer_dtovns.shap_values(choosen_instance_dtovns11)
shap.initjs()
shap.force_plot(explainer_dtovns.expected_value[1], shap_values_dtovns11[1], choosen_instance_dtovns11, 
                show=True, matplotlib=True)#, figsize=[27,2.5]) 
# OR standardised - Feature Importance Plot: Global Interpretability
shap_values_dtor = shap.TreeExplainer(dtmodel_or).shap_values(Xsel_train)
shap.summary_plot(shap_values_dtor, Xsel_train, plot_type="bar")
# OR standardised - Individual SHAP Value Plot — Local Interpretability 
X_output_dtor = Xsel_test.copy()
X_output_dtor.loc[:,'predict'] = np.round(dtmodel_or.predict(X_output_dtor),2)
explainer_dtor = shap.TreeExplainer(dtmodel_or)
choosen_instance_dtor28 = X_output_dtor.loc[[28]] # positive case example
shap_values_dtor28 = explainer_dtor.shap_values(choosen_instance_dtor28)
shap.initjs()
shap.force_plot(explainer_dtor.expected_value[1], shap_values_dtor28[1], choosen_instance_dtor28, 
                show=True, matplotlib=True, figsize=[27,2.5]) 
choosen_instance_dtor10 = X_output_dtor.loc[[10]] # negative case example
shap_values_dtor10 = explainer_dtor.shap_values(choosen_instance_dtor10)
shap.initjs()
shap.force_plot(explainer_dtor.expected_value[1], shap_values_dtor10[1], choosen_instance_dtor10, 
                show=True, matplotlib=True, figsize=[24,2.5])
# OR non-standardised - Feature Importance Plot: Global Interpretability
shap_values_dtorns = shap.TreeExplainer(dtmodelns_or).shap_values(Xselns_train)
shap.summary_plot(shap_values_dtorns, Xselns_train, plot_type="bar")
# OR non- standardised - Individual SHAP Value Plot — Local Interpretability 
X_output_dtorns = Xselns_test.copy()
X_output_dtorns.loc[:,'predict'] = np.round(dtmodelns_or.predict(X_output_dtorns),2)
explainer_dtorns = shap.TreeExplainer(dtmodelns_or)
choosen_instance_dtorns15 = X_output_dtorns.loc[[15]] # positive case example
shap_values_dtorns15 = explainer_dtorns.shap_values(choosen_instance_dtorns15)
shap.initjs()
shap.force_plot(explainer_dtorns.expected_value[1], shap_values_dtorns15[1], choosen_instance_dtorns15, 
                show=True, matplotlib=True) 
choosen_instance_dtorns10 = X_output_dtorns.loc[[10]] # negative case example
shap_values_dtorns10 = explainer_dtorns.shap_values(choosen_instance_dtorns10)
shap.initjs()
shap.force_plot(explainer_dtorns.expected_value[1], shap_values_dtorns10[1], choosen_instance_dtorns10, 
                show=True, matplotlib=True)

