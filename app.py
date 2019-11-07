from flask import Flask, render_template, request, redirect, session, Response
import requests
import simplejson as json
from bokeh.plotting import figure, show
from bokeh.resources import CDN
from bokeh.embed import components, file_html
import pandas as pd
import sklearn as skl
import dill
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


app = Flask(__name__)
app.secret_key = 'jabberwocky'

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

@app.route('/plot1.png')
def plot1_png():


    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    data = pd.read_csv('data.csv',delimiter = ',', parse_dates=['Date'],date_parser = parse_year)
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
    
    
    ArmedAndFatal = np.sum(np.logical_and(data.Fatal == 'F',data.SubjectArmed == 'Y'))
    ArmedAndFatal += np.sum(np.logical_and(data2.ARMED_WITH != 'None',data2.CASUALTY == 'Deceased'))
    UnarmedAndFatal = np.sum(np.logical_and(data.Fatal == 'F',data.SubjectArmed == 'N'))
    UnarmedAndFatal += np.sum(np.logical_and(data2.ARMED_WITH == 'None',data2.CASUALTY == 'Deceased'))
    ArmedAndNonfatal = np.sum(np.logical_and(data.Fatal == 'N',data.SubjectArmed == 'Y'))
    ArmedAndNonfatal += np.sum(np.logical_and(data2.ARMED_WITH != 'None',data2.CASUALTY != 'Deceased'))
    UnarmedAndNonfatal = np.sum(np.logical_and(data.Fatal == 'N',data.SubjectArmed == 'N'))
    UnarmedAndNonfatal += np.sum(np.logical_and(data2.ARMED_WITH == 'None',data2.CASUALTY != 'Deceased'))
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)               
    label = ['Armed,Fatally shot','Armed,Nonfatally shot','Unarmed,Fatally shot','Unarmed,Nonfatally shot']                
    p1 = ax.bar(np.arange(2),[ArmedAndFatal,UnarmedAndFatal],0.5,color=(0,0,0))
    p2 = ax.bar(np.arange(2),[ArmedAndNonfatal,UnarmedAndNonfatal],0.5,color=(0.5,0.5,0.5),bottom=[ArmedAndFatal,UnarmedAndFatal])
    ax.set_ylabel('Officer-Involved Shootings')
    ax.set_xlabel('Armed Status')
    ax.set_title('Officer-Involved Shootings by Subject Armed Status')       
    ax.set_xticks([0,1])
    ax.set_xticklabels(('Armed','Unarmed'))   
    ax.legend(['Fatally shot','Nonfatally shot'])
    return fig
    
@app.route('/plot2.png')
def plot2_png():
    fig = create_figure2()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure2():
    data = pd.read_csv('data.csv',delimiter = ',', parse_dates=['Date'],date_parser = parse_year)
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
    
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1) 
    p1 = ax.bar(x,y3,0.5,color=(0,0,0))
    p2 = ax.bar(x,y4,0.5,color=(0.5,0.5,0.5),bottom = y3)
    ax.set_ylabel('Officer-Involved Shootings')
    ax.set_xlabel('Subject Race')
    ax.set_title('Officer-Involved Shootings by Subject Race')
    ax.set_xticks(x)
    ax.set_xticklabels(FatallyShotByRace.keys())
    ax.legend(['Fatally shot','Nonfatally shot'])
    return fig    
    
@app.route('/plot3.png')
def plot3_png():
    fig = create_figure3()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure3():
    data = pd.read_csv('data.csv',delimiter = ',', parse_dates=['Date'],date_parser = parse_year)
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
    
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1) 
    x2 = range(0,8)
    p1 = ax.bar(x2,y5,0.5,color=(0,0,0))
    p2 = ax.bar(x2,y6,0.5,color=(0.5,0.5,0.5),bottom = y5)
    ax.set_ylabel('Officer-Involved Shootings')
    ax.set_xlabel('Officer Race')
    ax.set_title('Officer-Involved Shootings by Officer Race')
    ax.set_xticks(x2)
    ax.set_xticklabels(FatallyShotByOfficerRace.keys())
    ax.legend(['Fatally shot','Nonfatally shot'])
    return fig    

@app.route('/plot4.png')
def plot4_png():
    fig = create_figure4()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure4():
    data = pd.read_csv('data.csv',delimiter = ',', parse_dates=['Date'],date_parser = parse_year)
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
        
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1) 
    p1 = ax.bar([0,1,2],FatallyShotByGender,0.5,color=(0,0,0))
    p2 = ax.bar([0,1,2],NonfatallyShotByGender,0.5,color=(0.5,0.5,0.5),bottom=FatallyShotByGender)
    ax.set_ylabel('Officer-Involved Shootings')
    ax.set_xlabel('Subject Gender')
    ax.set_title('Officer-Involved Shootings by Subject Gender')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['Male','Female','Unknown'])
    ax.legend(['Fatally shot','Survived'])
    return fig    
    
@app.route('/plot5.png')
def plot5_png():
    fig = create_figure5()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure5():
    data = pd.read_csv('data.csv',delimiter = ',', parse_dates=['Date'],date_parser = parse_year)
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
 
    minAge = 10
    maxAge = 75
    FatallyShotByAge = np.zeros(maxAge-minAge)
    NonfatallyShotByAge = np.zeros(maxAge-minAge)
    for age in range(minAge,maxAge):
        FatallyShotByAge[age-minAge] = np.sum(np.logical_and(data.SubjectAge == str(age), data.Fatal == 'F'))
        NonfatallyShotByAge[age-minAge] = np.sum(np.logical_and(data.SubjectAge == str(age), data.Fatal == 'N'))
        
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1) 
    x3 = range(minAge,maxAge)
    p1 = ax.bar(x3,FatallyShotByAge,0.5,color=(0,0,0))
    p2 = ax.bar(x3,NonfatallyShotByAge,0.5,color=(0.5,0.5,0.5),bottom=FatallyShotByAge)
    ax.set_ylabel('Officer-Involved Shootings')
    ax.set_xlabel('Subject Age')
    ax.set_title('Officer-Involved Shootings by Subject Age')
    ax.legend(['Fatally shot','Nonfatally Shot'])
    return fig        

@app.route('/plot6.png')
def plot6_png():
    fig = create_figure6()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure6():
    data = pd.read_csv('data.csv',delimiter = ',', parse_dates=['Date'],date_parser = parse_year)
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

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1) 
    x4 = range(1,maxOfficers)
    p1 = ax.bar(x4,FatallyShotByNumOfficers,0.5,color=(0,0,0))
    p2 = ax.bar(x4,NonfatallyShotByNumOfficers,0.5,color=(0.5,0.5,0.5),bottom=FatallyShotByNumOfficers)
    ax.set_ylabel('Officer-Involved Shootings')
    ax.set_xlabel('Number of Officers Involved')
    ax.set_title('Officer-Involved Shootings by Officer Number')
    ax.legend(['Fatally shot','Nonfatally Shot'])
    return fig   

@app.route('/plot7.png')
def plot7_png():
    fig = create_figure7()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure7():
    data = pd.read_csv('data.csv',delimiter = ',', parse_dates=['Date'],date_parser = parse_year)
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
    data.OfficerRace = data.OfficerRace.str.replace('Other','O', regex=True)
    data.OfficerRace = data.OfficerRace.str.replace('[A-Z][;:/ ]','x', regex=True)
    data.OfficerRace = data.OfficerRace.str.replace('x[A-Z;:/ ]+','x', regex=True)
    data.OfficerRace = data.OfficerRace.str.replace('x+','Multiple', regex=True)
    data.OfficerRace = data.OfficerRace.str.replace('m/m','Multiple', regex=True)
    data.OfficerRace = data.OfficerRace.str.replace('[A-Z]Multiple','Multiple', regex=True)


    fig = Figure()
    
    x_data = data.OfficerRace
    y_data = data.SubjectRace
    
    c = Counter(zip(x_data,y_data))
    s = [2*c[(xx,yy)] for xx,yy in zip(x_data,y_data)]
    ax = fig.add_subplot(111)
    p1 = ax.scatter(x_data,y_data,s=s)
    ax.set_ylabel('Subject Race')
    ax.set_xlabel('Officer Race')
    ax.set_title('Frequency of shootings according to subject and officer race combination')
    return fig   


@app.route('/',methods=['GET','POST'])
def index():

  if request.method == 'GET':
    return render_template('index.html')
  else:
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():

    with open('combinedModel','rb') as in_strm:
        combinedModel = dill.load(in_strm)
        
    age = request.form['age']
    off_no = request.form['off_no']
    armed = request.form['armed']
    subj_race = request.form['subj_race']
    off_race = request.form['off_race']
    subj_gender = request.form['subj_gender']
    
    if age == '':
        age = 18
    if off_no == '':
        off_no = 1
    if armed == '':
        armed = 'Y'
    if subj_race == '':
        subj_race = 'W'
    if off_race == '':
        off_race = 'W'
    if subj_gender == '':
        subj_gender = 'M'
        
    temp = {'SubjectAge':age, 'NumberOfOfficers':off_no, 'SubjectArmed':armed, 'SubjectRace':subj_race, 'OfficerRace':off_race, 'SubjectGender':subj_gender}
    X = pd.DataFrame(temp, index=['Subject'])
    Y = combinedModel.predict(X)
    Y_proba = combinedModel.predict_proba(X)
    prob = round(max(Y_proba[0][0],Y_proba[0][1]),2)
    
    if Y == True:
        predicted_outcome = 'Nonfatal'
    else:
        predicted_outcome = 'Fatal'
        
    return render_template('prediction.html', outcome = predicted_outcome, probability=prob)

@app.route('/model',methods=['GET','POST'])
def model():
    return render_template('model.html')

@app.route('/conclusions',methods=['GET','POST'])
def conclusions():
    return render_template('conclusions.html')

if __name__ == '__main__':
  app.run(port=33507)
