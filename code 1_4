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

""" Functions"""

def grade_scatplot (string):
    data.plot(kind = 'scatter', x = string, y = 'num_grade')
    return

def grade_linreg(var):
    data_temp = data.loc[:,['num_grade',var]]
    data_temp = data_temp.dropna()
    mod = sm.OLS(data_temp['num_grade'],data_temp[var])
    res = mod.fit()
    print res.summary()
    return
    
    


"""Load in the data"""

data0711 = pd.read_csv("LoanStats3a.csv")
data1213 = pd.read_csv("LoanStats3b.csv")
data14 = pd.read_csv("LoanStats3c.csv")
data15 = pd.read_csv("LoanStats3d.csv")
data16Q1 = pd.read_csv("LoanStats_2016Q1.csv")
data16Q2 = pd.read_csv("LoanStats_2016Q2.csv")
data16Q3 = pd.read_csv("LoanStats_2016Q3.csv")




"""Concatenate the different dataframes"""

frame = [data0711, data1213, data14, data15, data16Q1, data16Q2, data16Q3]
data = pd.concat(frame, ignore_index = True)



"""Partition the training, development and test sets."""

""" We can partition by the following if the member ID is not assigned by some characteristics with
user with some of the characteristics assigned a member ID that is the multiple of 5"""

data['ASSIGN'] = data['member_id'] % 5+1    


"""Partition the data into test, development and training set"""

data_test = data[data.ASSIGN == 1]
data_dev = data[data.ASSIGN == 2]
data_train = data[(data.ASSIGN == 3) | (data.ASSIGN == 4) | (data.ASSIGN == 5)]

"""Output to csv"""
data_test.to_csv("data_test.csv")
data_dev.to_csv("data_dev.csv")
data_train.to_csv("data_train.csv")
data.to_csv("overall sorted data.csv")

"""Default rate????"""
number_default = len(data[data.loan_status == "Charged Off"])
"""How should we scope the current loans?"""


"""Converting categorical grading to numerical values"""
grade_lib = {'G':1, 'F':2, 'E':3, 'D':4, 'C':5, 'B':6, 'A':7}
num_grade = []
for i in data['sub_grade']:
   num_grade.append(grade_lib[i[0]]*5 - int(i[1]))
   
data['num_grade'] = pd.Series(num_grade, index = data.index)


"""Scatter plot of grade and other factors"""
# 1. avg_cur_bal
grade_scatplot ('avg_cur_bal')

# 2. chargeoff_within_12_mths
grade_scatplot('chargeoff_within_12_mths')

# 3. dti
data.plot(kind = 'scatter', x = 'dti', y = 'num_grade')

# 4. num_bc_sats      umber of satisfactory bankcard accounts
data.plot(kind = 'scatter', x = 'num_bc_sats', y = 'num_grade')

# 5. pub_rec
data.plot(kind = 'scatter', x = 'pub_rec', y = 'num_grade')

# 6. delinq_2yrs
data.plot(kind = 'scatter', x = 'delinq_2yrs', y = 'num_grade')

# 7. emp_length
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




###### HONESTLY NEED TO LOOK AT THE DENSITY DISTRIBUTION OF THE DATA POINT AT LEFT TAIL.



# 22. tot_cur_bal
data.plot(kind = 'scatter', x = 'tot_cur_bal', y = 'num_grade')
## perfect positive relationship

# 23. verification_status
data.plot(kind = 'scatter', x = 'verification_status', y = 'num_grade')









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
























