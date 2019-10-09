# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:45:51 2019

@author: Sean
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import base
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn import model_selection
from sklearn import metrics
import dill

def parse_year(date):
    
    date.strip()
    if len(date) == 4:
        year = date
    elif '/' in date:               
            tempYear = date[date.rfind('/')+1:]
            if len(tempYear) == 2:
                year = int('20'+tempYear)
                 
            else:
                year = tempYear
    elif '-' in date:
            year = int(date[0:4])            
    return pd.datetime(int(year),1,1)  

plots = False
data = pd.read_csv('ViceNews_FullOISData - Sheet1.csv',delimiter = ',', parse_dates=['Date'],date_parser = parse_year)
data2 = pd.read_csv('data2.csv',delimiter=',')

#Data sanitization
data.SubjectRace = data.SubjectRace.fillna('U')
data.OfficerRace = data.OfficerRace.fillna('U')
data.SubjectGender = data.SubjectGender.fillna('U')
data.SubjectArmed = data.SubjectArmed.fillna('U')
data.OfficerRace = data.OfficerRace.str.replace('Unknown','U', regex=True)
data.OfficerRace = data.OfficerRace.str.replace('WHITE','W', regex=True)
data.OfficerRace = data.OfficerRace.str.replace('BLACK','B', regex=True)
data.OfficerRace = data.OfficerRace.str.replace('ASIAN','A', regex=True)
data.OfficerRace = data.OfficerRace.str.replace('Multi-Racial','U', regex=True)
#Sanitize to take maximum age estimate
data.SubjectAge = data.SubjectAge.str.replace('[0-9][0-9]-','', regex=True)
data.SubjectAge = data.SubjectAge.str.replace('[0-9]-','', regex=True)
data.SubjectAge = data.SubjectAge.str.replace('UNKNOWN','0', regex=True)
data.SubjectAge = data.SubjectAge.str.replace('Juvenile','0', regex=True)
data.SubjectAge = data.SubjectAge.str.replace('U','0', regex=True)
data.SubjectAge = data.SubjectAge.fillna(0)

with open("data","wb") as dill_file:
    dill.dump(data, dill_file)


ArmedAndFatal = np.sum(np.logical_and(data.Fatal == 'F',data.SubjectArmed == 'Y'))
ArmedAndFatal += np.sum(np.logical_and(data2.ARMED_WITH != 'None',data2.CASUALTY == 'Deceased'))
UnarmedAndFatal = np.sum(np.logical_and(data.Fatal == 'F',data.SubjectArmed == 'N'))
UnarmedAndFatal += np.sum(np.logical_and(data2.ARMED_WITH == 'None',data2.CASUALTY == 'Deceased'))
ArmedAndNonfatal = np.sum(np.logical_and(data.Fatal == 'N',data.SubjectArmed == 'Y'))
ArmedAndNonfatal += np.sum(np.logical_and(data2.ARMED_WITH != 'None',data2.CASUALTY != 'Deceased'))
UnarmedAndNonfatal = np.sum(np.logical_and(data.Fatal == 'N',data.SubjectArmed == 'N'))
UnarmedAndNonfatal += np.sum(np.logical_and(data2.ARMED_WITH == 'None',data2.CASUALTY != 'Deceased'))

x = np.zeros(7)
y1 = np.zeros(7)
y2 = np.zeros(7)
FatallyShotByYear = np.zeros(7)
NonfatallyShotByYear = np.zeros(7)
i=0
for year in range(2010,2017):
    FatallyShotByYear[i] = np.sum(np.logical_and(data.Date == pd.datetime(int(year),1,1), data.Fatal == 'F'))
    NonfatallyShotByYear[i] = np.sum(np.logical_and(data.Date == pd.datetime(int(year),1,1), data.Fatal == 'N'))
    x[i] = int(year)
    y1[i] = FatallyShotByYear[i]
    y2[i] = NonfatallyShotByYear[i]
    i = i+1

x = range(0,7)
FatallyShotByRace = dict()
NonfatallyShotByRace = dict()
y3 = np.zeros(7)
y4 = np.zeros(7)
i=0
for race in ['W','L','B','U','A','O']:
    FatallyShotByRace[race] = np.sum(np.logical_and(data.SubjectRace == race, data.Fatal == 'F'))
    FatallyShotByRace[race] += np.sum(np.logical_and(data2.RACE == race, data2.CASUALTY == 'Deceased'))
    NonfatallyShotByRace[race] = np.sum(np.logical_and(data.SubjectRace == race, data.Fatal == 'F'))
    NonfatallyShotByRace[race] += np.sum(np.logical_and(data2.RACE == race, data2.CASUALTY != 'Deceased'))
    y3[i] = float(FatallyShotByRace[race])
    y4[i] = float(NonfatallyShotByRace[race])
    i+=1    
   
x = range(0,7)    
FatallyShotByOfficerRace = dict()
NonfatallyShotByOfficerRace = dict()
y5 = np.zeros(8)
y6 = np.zeros(8)
i=0
for race in ['W','L','B','U','A','O','H']:
    FatallyShotByOfficerRace[race] = np.sum(np.logical_and(data.OfficerRace == race, data.Fatal == 'F'))
    NonfatallyShotByOfficerRace[race] = np.sum(np.logical_and(data.OfficerRace == race, data.Fatal == 'N'))
    y5[i] = float(FatallyShotByOfficerRace[race])
    y6[i] = float(NonfatallyShotByOfficerRace[race])
    i+=1       
    
FatallyShotByOfficerRace['Multiple'] = np.sum(np.logical_and(data.OfficerRace.str.find(';') != -1, data.Fatal == 'F'))   
NonfatallyShotByOfficerRace['Multiple'] = np.sum(np.logical_and(data.OfficerRace.str.find(';') != -1, data.Fatal == 'N'))  
y5[7] = float(FatallyShotByOfficerRace['Multiple']) 
y6[7] = float(NonfatallyShotByOfficerRace['Multiple'])

FatallyShotByGender = np.zeros(3)
NonfatallyShotByGender = np.zeros(3)    

FatallyShotByGender[0] = np.sum(np.logical_and(data.Fatal == 'F',data.SubjectGender == 'M'))   
FatallyShotByGender[0] += np.sum(np.logical_and(data2.CASUALTY == 'Deceased',data2.GENDER == 'M'))   
FatallyShotByGender[1] = np.sum(np.logical_and(data.Fatal == 'F',data.SubjectGender == 'F')) 
FatallyShotByGender[1] += np.sum(np.logical_and(data2.CASUALTY == 'Deceased',data2.GENDER == 'F'))    
FatallyShotByGender[2] = np.sum(np.logical_and(data.Fatal == 'F',data.SubjectGender == 'U'))         
NonfatallyShotByGender[0] = np.sum(np.logical_and(data.Fatal == 'N',data.SubjectGender == 'M'))   
NonfatallyShotByGender[0] += np.sum(np.logical_and(data2.CASUALTY != 'Deceased',data2.GENDER == 'M'))  
NonfatallyShotByGender[1] = np.sum(np.logical_and(data.Fatal == 'N',data.SubjectGender == 'F'))
NonfatallyShotByGender[1] += np.sum(np.logical_and(data2.CASUALTY != 'Deceased',data2.GENDER == 'F'))   
NonfatallyShotByGender[2] = np.sum(np.logical_and(data.Fatal == 'N',data.SubjectGender == 'U')) 



minAge = 10
maxAge = 75
FatallyShotByAge = np.zeros(maxAge-minAge)
NonfatallyShotByAge = np.zeros(maxAge-minAge)
for age in range(minAge,maxAge):
    FatallyShotByAge[age-minAge] = np.sum(np.logical_and(data.SubjectAge == str(age), data.Fatal == 'F'))
    NonfatallyShotByAge[age-minAge] = np.sum(np.logical_and(data.SubjectAge == str(age), data.Fatal == 'N'))

data.NumberOfOfficers = data.NumberOfOfficers.str.replace(' or More','', regex=True)
data.NumberOfOfficers = data.NumberOfOfficers.str.replace('>','', regex=True)
data.NumberOfOfficers = data.NumberOfOfficers.str.replace('U','0', regex=True)
data.NumberOfOfficers = data.NumberOfOfficers.fillna(0)

maxOfficers = 10
FatallyShotByNumOfficers = np.zeros(maxOfficers-1)
NonfatallyShotByNumOfficers = np.zeros(maxOfficers-1)
for numOfficers in range(1,maxOfficers):
    FatallyShotByNumOfficers[numOfficers-1] = np.sum(np.logical_and(data.NumberOfOfficers == str(numOfficers), data.Fatal == 'F'))
    NonfatallyShotByNumOfficers[numOfficers-1] = np.sum(np.logical_and(data.NumberOfOfficers == str(numOfficers), data.Fatal == 'N'))

      
if (plots == True):
    mpl_fig1 = mpl.pyplot.figure()
    ax = mpl_fig1.add_subplot(111)                
    label = ['Armed,Fatally shot','Armed,Nonfatally shot','Unarmed,Fatally shot','Unarmed,Nonfatally shot']                
    p1 = ax.bar(np.arange(2),[ArmedAndFatal,UnarmedAndFatal],0.5,color=(0,0,0))
    p2 = ax.bar(np.arange(2),[ArmedAndNonfatal,UnarmedAndNonfatal],0.5,color=(0.5,0.5,0.5),bottom=[ArmedAndFatal,UnarmedAndFatal])
    ax.set_ylabel('Subjects')
    ax.set_xlabel('Status')
    ax.set_title('Comparison of armed and unarmed OIS subjects')       
    ax.set_xticks([0,1])
    ax.set_xticklabels(('Armed','Unarmed'))   
    ax.legend(['Fatally shot','Nonfatally shot'])
    
    mpl_fig2 = mpl.pyplot.figure()
    ax = mpl_fig2.add_subplot(111)
    p1 = ax.plot(x,y2/y1,color=(0,0,0))
    ax.set_ylabel('Shootings')
    ax.set_xlabel('Year')
    ax.set_title('Ratio of nonfatal:fatal officer-involved shootings by year')
    
    mpl_fig3 = mpl.pyplot.figure()
    ax = mpl_fig3.add_subplot(111)
    p1 = ax.bar(x,y3,0.5,color=(0,0,0))
    p2 = ax.bar(x,y4,0.5,color=(0.5,0.5,0.5),bottom = y3)
    ax.set_ylabel('Shootings')
    ax.set_xlabel('Subject Race')
    ax.set_title('Shootings by Subject Race')
    ax.set_xticks(x)
    ax.set_xticklabels(FatallyShotByRace.keys())
    ax.legend(['Fatally shot','Nonfatally shot'])
    
    mpl_fig4 = mpl.pyplot.figure()
    ax = mpl_fig4.add_subplot(111)
    x2 = range(0,8)
    p1 = ax.bar(x2,y5,0.5,color=(0,0,0))
    p2 = ax.bar(x2,y6,0.5,color=(0.5,0.5,0.5),bottom = y5)
    ax.set_ylabel('Shootings')
    ax.set_xlabel('Officer Race')
    ax.set_title('Shootings by Officer Race')
    ax.set_xticks(x2)
    ax.set_xticklabels(FatallyShotByOfficerRace.keys())
    ax.legend(['Fatally shot','Nonfatally shot'])
    
    mpl_fig5= mpl.pyplot.figure()
    ax = mpl_fig5.add_subplot(111)
    p1 = ax.bar([0,1,2],FatallyShotByGender,0.5,color=(0,0,0))
    p2 = ax.bar([0,1,2],NonfatallyShotByGender,0.5,color=(0.5,0.5,0.5),bottom=FatallyShotByGender)
    ax.set_ylabel('Shootings')
    ax.set_xlabel('Subject Gender')
    ax.set_title('Shootings by Subject Gender')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['Male','Female','Unknown'])
    ax.legend(['Fatally shot','Survived'])
    
    x3 = range(minAge,maxAge)
    mpl_fig6= mpl.pyplot.figure()
    ax = mpl_fig6.add_subplot(111)
    p1 = ax.bar(x3,FatallyShotByAge,0.5,color=(0,0,0))
    p2 = ax.bar(x3,NonfatallyShotByAge,0.5,color=(0.5,0.5,0.5),bottom=FatallyShotByAge)
    ax.set_ylabel('Shootings')
    ax.set_xlabel('Subject Age')
    ax.set_title('Shootings by Subject Age')
    ax.legend(['Fatally shot','Nonfatally Shot'])
    
    x3 = range(minAge,maxAge)
    mpl_fig7= mpl.pyplot.figure()
    ax = mpl_fig7.add_subplot(111)
    p1 = ax.plot(x3,NonfatallyShotByAge / FatallyShotByAge,color=(0,0,0))
    ax.set_ylabel('Shootings')
    ax.set_xlabel('Subject Age')
    ax.set_title('Ratio of Nonfatal:fatal Shootings by Subject Age')
    
    x4 = range(1,maxOfficers)
    mpl_fig7= mpl.pyplot.figure()
    ax = mpl_fig7.add_subplot(111)
    p1 = ax.bar(x4,FatallyShotByNumOfficers,0.5,color=(0,0,0))
    p2 = ax.bar(x4,NonfatallyShotByNumOfficers,0.5,color=(0.5,0.5,0.5),bottom=FatallyShotByNumOfficers)
    ax.set_ylabel('Shootings')
    ax.set_xlabel('Number of Officers Involved')
    ax.set_title('Shootings by Officer Number')
    ax.legend(['Fatally shot','Nonfatally Shot'])
    
    x4 = range(1,maxOfficers)
    mpl_fig8= mpl.pyplot.figure()
    ax = mpl_fig8.add_subplot(111)
    p1 = ax.plot(x4,NonfatallyShotByNumOfficers / FatallyShotByNumOfficers,color=(0,0,0))
    ax.set_ylabel('Shootings')
    ax.set_xlabel('Number of Officers Involved')
    ax.set_title('Ratio of Nonfatal:fatal Shootings by Officer Number')

boolFatal = (data.Fatal == 'N')
boolFatal = boolFatal.fillna(0)
ages = data.SubjectAge.astype('float')
ages = ages.fillna(0)

class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, estimator):
        # What needs to be done here?
        self.estimator = estimator
    
    def fit(self, X, y):
        # Fit the stored estimator.
        self.estimator.fit(X,y)
        # Question: what should be returned?
        return self
    
    def transform(self, X):
        # Use predict on the stored estimator as a "transformation".
        y0 = self.estimator.predict(X)
        
        output_list = list()        
        for y in y0:
            inner_list = list()
            inner_list.append(y)
            output_list.append(inner_list)
    #    np.reshape(y0,(-1,1))
        return output_list
        # Be sure to return a 2-D array.


class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self,col_names):
        self.col_names = col_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        series = X.loc[:,self.col_names]
        new_X = list()
        for row in series:
            entry = list()
            entry.append(row)
            new_X.append(entry)
        return new_X

class TfidfEncoder(base.BaseEstimator, base.TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        new_X = list()
        for row in X:
            new_X.append(row[0])
        return new_X
    
class CustomLabelEncoder(base.BaseEstimator, base.TransformerMixin):
    def fit(self,X,y=None, **fit_params):
        return self
    def transform(self,X):
        return LabelEncoder.fit(X).transform(X)
    
def categorize(X):
    categories = X.unique()
    output = list()
    intermediate = list()
    for category in categories:
        intermediate.append(category)
    output.append(intermediate)
    return output

cs = ColumnSelectTransformer('SubjectAge')
cs2 = ColumnSelectTransformer('SubjectArmed')
cs3 = ColumnSelectTransformer('NumberOfOfficers')
te = TfidfEncoder()
test = te.fit_transform(cs.fit_transform(data))
test2 = te.fit_transform(cs2.fit_transform(data))
test3 = te.fit_transform(cs3.fit_transform(data))

test4 = categorize(data.OfficerRace)
oe = OrdinalEncoder(categorize(data.OfficerRace))


armed_est = Pipeline([
                    ("armed",ColumnSelectTransformer('SubjectArmed')),
                    ("vec_armed",OrdinalEncoder(categories = categorize(data.SubjectArmed))),
        ])    

subj_race_est = Pipeline([
                ("armed",ColumnSelectTransformer('SubjectRace')),
                ("vec_armed",OrdinalEncoder(categories = categorize(data.SubjectRace))),
    ])  

off_race_est = Pipeline([
                ("armed",ColumnSelectTransformer('OfficerRace')),
                ("vec_armed",OrdinalEncoder(categories = categorize(data.OfficerRace))),
    ])  
    
gender_est = Pipeline([
                ("armed",ColumnSelectTransformer('SubjectGender')),
                ("vec_armed",OrdinalEncoder(categories = categorize(data.SubjectGender))),
    ])      
        

combinedFeatures = FeatureUnion([
                            ("age",ColumnSelectTransformer('SubjectAge')),
                            ("no",ColumnSelectTransformer('NumberOfOfficers')),                           
                            ("armed",armed_est),
                            ("s_race",subj_race_est),
                            ("o_race",off_race_est),
                            ("gender",gender_est)
                            ])
 
combinedModel = Pipeline([
                    ("features",combinedFeatures),
                    ("model",RandomForestClassifier(n_estimators=50,max_depth=10))
        ])   
combinedModel.fit(data,boolFatal)
y_pred = combinedModel.predict(data)
acc = accuracy_score(boolFatal,y_pred)
prec = precision_score(boolFatal,y_pred)
print("Accuracy:" + str(acc) + "\nPrecision:" + str(prec))
 
X_train, X_test, y_train, y_test = train_test_split(data,boolFatal)


combinedModel.fit(X_train,y_train)
y_known = combinedModel.predict(X_train)
#print(balanced_accuracy_score(y_train,y_known))
y_pred = combinedModel.predict(X_test)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
print("Model accuracy: " + str(acc) + "\nPrecision:" + str(prec))

indices = np.random.permutation(range(len(boolFatal)))
X_random_order, y_random_order = data.reindex(indices), boolFatal[indices]
cv_test_errors = []

#with open("combinedModel","wb") as dill_file:
#    dill.dump(combinedModel, dill_file)

#params = range(250,350,10)
#for param in params:
#    est = combinedModel
#    
#    est.set_params(model__n_estimators = param)
#    cv_test_metric = model_selection.cross_val_score(
#            est,
#            X_random_order,
#            y_random_order,
#            cv = 5,
#            scoring = 'accuracy')
#    cv_test_errors.append(cv_test_metric.mean())
    
#print(cv_test_errors)    

gs = model_selection.GridSearchCV(
        combinedModel,
        {"model__n_estimators":range(10,250,20), "model__max_depth":range(2,20,2)},
        cv=5,
        n_jobs=2,
        scoring='accuracy'
        )
gs.fit(X_random_order,y_random_order)
print(gs.best_params_)

X_train, X_test, y_train, y_test = train_test_split(data,boolFatal)
naiveprediction = np.ones(len(y_test))
naivescore = accuracy_score(naiveprediction,y_test)
naiveprec = precision_score(naiveprediction,y_test)
print("Naive accuracy:" + str(naivescore) + "\nNaive precision:" + str(naiveprec))

modelObj = combinedModel.named_steps['model']
importances = modelObj.feature_importances_
std = np.std([tree.feature_importances_ for tree in modelObj.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(0,6), importances[indices],
       color="r", yerr=std[indices], align="center")
#plt.xticks(range(0,6), indices)
plt.xlim([-1, 6])
plt.xticks(range(0,6),['Age','O. Race','O. Count','Race','Armed','Gender'])
plt.show()

age = '18'
off_no = '1'
armed = 'Y'
subj_race = 'W'
off_race = 'W'
subj_gender = 'M'

temp = {'SubjectAge':age, 'NumberOfOfficers':off_no, 'SubjectArmed':armed, 'SubjectRace':subj_race, 'OfficerRace':off_race, 'SubjectGender':subj_gender}

X = pd.DataFrame(temp, index=['Subject'])

Y = combinedModel.predict(X)
print(Y)