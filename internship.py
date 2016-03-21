import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn import ensemble, preprocessing, cross_validation
from sklearn import metrics 
from sklearn import grid_search, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


if __name__ == '__main__':
    train = pd.read_csv('C:\\Users\\Nj_neeraj\\Documents\\Data_Science\\bigmart\\dateyourdata\\train.csv',
                        parse_dates= 'Earliest_Start_Date')#,nrows = 300)
    test = pd.read_csv('C:\\Users\\Nj_neeraj\\Documents\\Data_Science\\bigmart\\dateyourdata\\test.csv',
                       parse_dates= 'Earliest_Start_Date' )#,nrows = 300)
    internship = pd.read_csv('C:\\Users\\Nj_neeraj\\Documents\\Data_Science\\bigmart\\dateyourdata\\Internship.csv',
                             parse_dates= ['Start_Date','Internship_deadline'])#,nrows = 300)
    student = pd.read_csv('C:\\Users\\Nj_neeraj\\Documents\\Data_Science\\bigmart\\dateyourdata\\student.csv')
    

    internship  = internship[internship.columns[0:13]]
    train = train.merge(internship, on = 'Internship_ID' ,how = 'left')
    test = test.merge(internship, on = 'Internship_ID' ,how = 'left')
    
    int_id = test.Internship_ID
    st_id = test.Student_ID
    
    
    

    col = ['Student_ID', 'Institute_Category', 'Institute_location', 'hometown', 'Degree', 'Stream', 'Current_year',
           'Year_of_graduation', 'Performance_PG', 'PG_scale','Performance_UG', 'UG_Scale', 'Performance_12th',
           'Performance_10th']
    
    c = student.groupby(col).agg('count').reset_index()    
    
    student = c[col]
    
    train = train.merge(student ,on = 'Student_ID', how = 'left' )
    test = test.merge(student ,on = 'Student_ID', how = 'left' )

   

    text_columns = []

    for f in train.columns:
        if train[f].dtype=='object':
            if f != 'loca':    
                text_columns.append(f)            
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(train[f].values) + list(test[f].values))
                train[f] = lbl.transform(list(train[f].values))
                test[f] = lbl.transform(list(test[f].values))
     
         
    train.replace(np.nan , -1 ,inplace = True)
    test.replace(np.nan , -1 ,inplace = True)
    
    target = train.Is_Shortlisted.values
    #train = train.drop(['Earliest_Start_Date','Start_Date','Internship_deadline'],axis =1)
    train = train.drop(['Is_Shortlisted','Earliest_Start_Date','Start_Date','Internship_deadline'],axis =1) 
    test = test.drop(['Earliest_Start_Date','Start_Date','Internship_deadline'],axis =1)
    #train.to_csv('C:\\Users\\Nj_neeraj\\Documents\\Data_Science\\bigmart\\dateyourdata\\trainforh2o.csv',index = False)
    #test.to_csv('C:\\Users\\Nj_neeraj\\Documents\\Data_Science\\bigmart\\dateyourdata\\testforh2o.csv',index = False)
    k = train.columns
    param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [80, 160, 320, 400]}
    gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
    gs.fit(train, target)

