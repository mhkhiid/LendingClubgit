import re
import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import os

matplotlib.style.use('ggplot')

def grade_scatplot (data, feature):
    data.plot(kind = 'scatter', x = feature, y = 'grade')
    plt.savefig(feature + '_grade.png')
    return


def grade_linreg(data,var):
    data_temp = data.loc[:,['grade',var]]
    data_temp = data_temp.dropna()
    mod = sm.OLS(data_temp['grade'],data_temp[var])
    res = mod.fit()
    print res.summary()
    return
    

def merge_data(files):

    data_merge = []
    for datafile in files:
        data_year = pd.read_csv(datafile)
        data_merge.append(data_year)

    data = pd.concat(data_merge, ignore_index = True)
    return data

def get_default_rate(data):
    num_default = len(data[data.loan_status == "Charged Off"])
    return float(num_default) / len(data)


def preprocessing(data):
    # Converting categorical grading to numerical values
    grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    verification_map = {'Not Verified':0, 'Source Verified':1, 'Verified':2}

    data = data.replace({'grade': grade_map, 'verification_status': verification_map})

    # substitute '< 1 years' to '1', '2 years' to '2', '10+ years' to '10'
    data['emp_length'] = pd.to_numeric(data.emp_length.str.replace(r'<? ?(\d+)[+]? year[s]?', r'\1').replace('n/a', np.NaN))

    data['term'] = pd.to_numeric(data.term.str.replace(' months', ''))

    # Deleting % in int_rate
    data['int_rate'] = pd.to_numeric(data.int_rate.str.replace('%', ''))
    
    
    # Change home_ownership to dummy variables   note: multicollinearity
    data_home = pd.get_dummies(data['home_ownership'])
    data_home.columns = "home_" + data_home.columns
    data = pd.concat([data, data_home],1)
    
    # Change verification_status to dummy variables
    data_veri = pd.get_dummies(data['verification_status'])
    #data_veri.columns = "veri_" + data_veri.columns
    data = pd.concat([data,data_veri],1)
    
    # Change payment_plan (default: n = 0, y = 1)
    data = data.replace({'payment_plan':{'n': 0 , 'y': 1}})
    
    # Change purpose to multiple dummy variables
    data_purpose = pd.get_dummies(data['purpose'])
    #data_purpose.columns = "purpose_" + data_purpose.columns
    data = pd.concat([data,data_purpose], 1)
    
    # Delete the % in revol_util
    data['revol_util'] = pd.to_numeric(data.revol_util.str.replace('%',''))
    
    # Convert initial_list_status to multiple dummy variables (multicollinearity not controlled)
    data_int = pd.get_dummies(data['initial_list_status'])
    #data_int.columns = "initial_list_" + data_int.columns
    data = pd.concat([data,data_int], 1)
    
    
    # Convert application_type to multiple dummy variables (multicollinearity not controlled)
    data_app = pd.get_dummies(data['application_type'])
    # data_app.columns = ['application_type_direct','application_type_individual', 'application_type_joint']
    data = pd.concat([data, data_app], 1)
    
    
    # Convert verification_status_joint to multiple dummy variables (multicollinearity automatically avoided since get_dummies does not account for NaN
    data_temp = pd.get_dummies(data['verification_status_joint'])
    data = pd.concat([data,data_temp],1)
    o_names = ["Not Verified", "Source Verified", "Verified"]
    s_names = ['Vefirication_Joint_Not','Verification_Joint_SVerified', 'Verification_Joint_Verified']
    for num in range(0,3):
       data.rename(columns = {o_names[num]:s_names[num]}, inplace = True)
    
    return data

    
def split_data(data):
    # Shuffle the data set to get training, development and test sets.
    data = data.reindex(np.random.permutation(data.index))
    data.reset_index(drop=True, inplace=True)

    num_data = len(data)
    batch = int(num_data / 5)

    data_test = data[0:batch]
    data_dev = data[batch:2*batch]
    data_train = data[2*batch:]
    data_dev.reset_index(drop=True, inplace=True)
    data_train.reset_index(drop=True, inplace=True)

    # Save to csv
    data_test.to_csv("data_test.csv")
    data_dev.to_csv("data_dev.csv")
    data_train.to_csv("data_train.csv")
    data.to_csv("overall sorted data.csv")
    return

    
def run():
    #files = [ "LoanStats3a.csv", "LoanStats3b.csv", "LoanStats3c.csv", "LoanStats3d.csv", "LoanStats_2016Q1.csv", "LoanStats_2016Q2.csv", "LoanStats_2016Q3.csv" ]
    #files = [ "LoanStats3a.csv" ]
    #data = merge_data(files)
    #data = preprocessing(data)
    #split_dataset(files)

    data_train = pd.read_csv("data_train.csv")
    default_rate = get_default_rate(data_train)
    print "Default rate is %f" % default_rate

    data_train = preprocessing(data_train)

    # Exploratory data analysis
    features = [ 'avg_cur_bal', 'chargeoff_within_12_mths', 'dti', 'num_bc_sats', 
                 'pub_rec', 'delinq_2yrs', 'emp_length', 'il_util', 'inq_fi', 
                 'inq_last_12m', 'max_bal_bc', 'mths_since_last_delinq', 
                 'mths_since_last_major_derog', 'num_accts_ever_120_pd', 
                 'num_actv_bc_tl', 'num_bc_sats', 'num_tl_120dpd_2m', 
                 'num_tl_30dpd', 'open_il_6m', 'pct_tl_nvr_dlq', 'pub_rec', 
                 'tot_cur_bal', 'verification_status' ]

    for i in features:
        grade_scatplot(data_train, i)

    # Linear regression
    grade_linreg('avg_cur_bal')

'''
Findings:
 
 inq_fi                          No obvious relationship
 inq_last_12m                    No obvious relationship
 max_bal_bc                      quite obvious positive relationship
 mths_since_last_delinq 
 mths_since_last_major_derog     Not so obvious
 num_accts_ever_120_pd           Obvious positive relationship???
 num_actv_bc_tl                  No obvious relationship
 num_bc_sats                     good relationship
 num_tl_120dpd_2m                good negative relationship, but sample space is too small
 num_tl_30dpd                    No relationship 
 open_il_6m                      positive relationship 
 pct_tl_nvr_dlq                  WHY MOST OF THE SCATTER PLOTS ARE > SHAPED??
 pub_rec                         counter-intuitive shape. 
 '''
 
"""
if __name__ == '__main__':
    run()
"""     
     
     
     
     
     
     
     
     
     
     
     
 
 