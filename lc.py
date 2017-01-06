# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 01:09:09 2016

@author: dingh
"""

import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

matplotlib.style.use('ggplot')

def grade_scatplot (data, feature):
    data.plot(kind = 'scatter', x = feature, y = 'grade')
    return


def grade_linreg(var):
    data_temp = data.loc[:,['grade',var]]
    data_temp = data_temp.dropna()
    mod = sm.OLS(data_temp['grade'],data_temp[var])
    res = mod.fit()
    print res.summary()
    return
    

def split_dataset(files):

    data_merge = []
    for datafile in files:
        data_year = pd.read_csv(datafile)
        data_merge.append(data_year)

    data = pd.concat(data_merge, ignore_index = True)

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


def get_default_rate(data):
    num_default = len(data[data.loan_status == "Charged Off"])
    return float(num_default) / len(data)


def preprocessing(data):
    # Converting categorical grading to numerical values
    grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    return data.replace({'grade': grade_map})


def run():
    #files = [ "LoanStats3a.csv", "LoanStats3b.csv", "LoanStats3c.csv", "LoanStats3d.csv", "LoanStats_2016Q1.csv", "LoanStats_2016Q2.csv", "LoanStats_2016Q3.csv" ]
    #files = [ "LoanStats3a.csv" ]
    #split_dataset(files)

    data_train = pd.read_csv("data_train.csv")
    default_rate = get_default_rate(data_train)
    print "Default rate is %f" % default_rate

    data_train = preprocessing(data_train)

    for i in [ 'avg_cur_bal', 'chargeoff_within_12_mths', 'dti', 'num_bc_sats', 'pub_rec', 'delinq_2yrs', 'emp_length']:
        grade_scatplot(data_train, i)


# 7. emp_length
## TODO
"""temp = []
for elem in data[data.emp_length == '10+ years']['emp_length']:
    temp.append(10)
data[data.emp_length == '10+ years'] = temp

temp = []
for elem in data[data.emp_length != 10]['emp_length']:
    temp.append(elem[0:1])

temp_mod = []
for elem in temp:
    if elem == 'n':
        elem = 0
    if elem == '<':
        elem = 0.5
    temp_mod.append(elem)   
        
data[data.emp_length != 10]['emp_length'] = temp_mod
# CANNOT IDENTIFY THE BUG HERE"""

# 8. il_util, 'inq_fi', 'inq_last_12m', 'max_bal_bc', 'mths_since_last_delinq, 'mths_since_last_major_derog', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_bc_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'open_il_6m', 'pct_tl_nvr_dlq', 'pub_rec', 'tot_cur_bal', 'verification_status'

'''

# Exploratory linear regression

# 1. avg_cur_bal
grade_linreg('avg_cur_bal')

"""
# 2. chargeoff_within_12_mths
data.plot(kind = 'scatter', x = 'chargeoff_within_12_mths', y = 'num_grade')

# 3. dti
data.plot(kind = 'scatter', x = 'dti', y = 'num_grade')

# 4. num_bc_sats      umber of satisfactory bankcard accounts
data.plot(kind = 'scatter', x = 'num_bc_sats', y = 'num_grade')

# 5. pub_rec
data.plot(kind = 'scatter', x = 'pub_rec', y = 'num_grade')

# 6. delinq_2yrs
data.plot(kind = 'scatter', x = 'delinq_2yrs', y = 'num_grade')

# 7. emp_length
emp = []
for elem in data[data.emp_length == '10+ years']['emp_length']:
    temp.append(10)
data[data.emp_length == '10+ years'] = temp

temp = []
for elem in data[data.emp_length != 10]['emp_length']:
    temp.append(elem[0:1])

temp_mod = []
for elem in temp:
    if elem == 'n':
        elem = 0
    if elem == '<':
        elem = 0.5
    temp_mod.append(elem)   
        
data[data.emp_length != 10]['emp_length'] = temp_mod
# CANNOT IDENTIFY THE BUG HERE

# 8. il_util
data.plot(kind = 'scatter', x = 'il_util', y = 'num_grade')

# 9. inq_fi
data.plot(kind = 'scatter', x = 'inq_fi', y = 'num_grade')
## No obvious relationship

# 10. inq_last_12m
data.plot(kind = 'scatter', x = 'inq_last_12m', y = 'num_grade')
## No obvious relationship

# 11. max_bal_bc
data.plot(kind = 'scatter', x = 'max_bal_bc', y = 'num_grade')
## quite obvious positive relationship

# 12. mths_since_last_delinq
data.plot(kind = 'scatter', x = 'mths_since_last_delinq', y = 'num_grade')

# 13. mths_since_last_major_derog
data.plot(kind = 'scatter', x = 'mths_since_last_major_derog', y = 'num_grade')
## Not so obvious

# 14. num_accts_ever_120_pd
data.plot(kind = 'scatter', x = 'num_accts_ever_120_pd', y = 'num_grade')
## Obvious positive relationship???

# 15. num_actv_bc_tl
data.plot(kind = 'scatter', x = 'num_actv_bc_tl', y = 'num_grade')
## No obvious relationship

# 16. num_bc_sats
data.plot(kind = 'scatter', x = 'num_bc_sats', y = 'num_grade')
## good relationship

# 17. num_tl_120dpd_2m
data.plot(kind = 'scatter', x = 'num_tl_120dpd_2m', y = 'num_grade')
## good negative relationship, but sample space is too small

# 18. num_tl_30dpd
data.plot(kind = 'scatter', x = 'num_tl_30dpd', y = 'num_grade')
## No relationship 

# 19. open_il_6m
data.plot(kind = 'scatter', x = 'open_il_6m', y = 'num_grade')
## positive relationship 

# 20. pct_tl_nvr_dlq
data.plot(kind = 'scatter', x = 'pct_tl_nvr_dlq', y = 'num_grade')
##WHY MOST OF THE SCATTER PLOTS ARE > SHAPED??

# 21. pub_rec
data.plot(kind = 'scatter', x = 'pub_rec', y = 'num_grade')
## counter-intuitive shape. 
"""
'''

if __name__ == '__main__':
    run()
