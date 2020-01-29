#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Student Name : Markus Proesch
# Cohort       : 4

################################################################################
# Import Packages
################################################################################

# importing packages
import pandas as pd # data science essentials
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # enhanced data visualization
import statsmodels.formula.api as smf # explanatory model 
from sklearn.model_selection import train_test_split #train/test/split
from sklearn.linear_model import LinearRegression #Linear Regression
from sklearn.linear_model import Ridge #Ridge Regression
from sklearn.linear_model import Lasso #Lasso Regression
from sklearn.ensemble import GradientBoostingRegressor #Gradient Boosting Regressor
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.preprocessing import StandardScaler # standard scaler


################################################################################
# Load Data
################################################################################


original_df  = pd.read_excel('Apprentice Chef Dataset.xlsx')

# Named my dataset chef while working on it 
chef         = original_df


################################################################################
# Feature Engineering, Variable Selection and (optional) Dataset Standardization
################################################################################

# Developed a new variable presenting the avg. price per meal per customer
# which became very important for the model later
chef['AVG_PRICE_PER_MEAL'] = chef['REVENUE'] / chef['TOTAL_MEALS_ORDERED']

# Outliers thresholds determined based on the histograms and scatterplots
total_meals_ord_hi   = 180
total_meals_ord_lo   = 25
unique_meals_pur_hi  = 10
contact_w_custo_s_lo = 2
contact_w_custo_s_hi = 10
avg_time_per_site_hi = 210
cancel_bef_noon_hi   = 4
cancel_aft_noon_hi   = 2
mobile_log_lo        = 4
mobile_log_hi        = 7
pc_log_hi            = 3
weekly_plan_hi       = 15
late_deliv_hi        = 5
avg_prep_vid_time_hi = 220
avg_prep_vid_time_lo = 65
largest_order_hi     = 8
master_class_hi      = 2
avg_click_per_lo     = 7
total_photo_hi       = 300
avg_price_per_m_hi   = 50
revenue_hi           = 2200


## Developing threshold for outliers
# REVENUE 
chef['out_REVENUE']  = 0
condition_hi = chef.loc[0:,'out_REVENUE'][chef['REVENUE'] 
                                          > revenue_hi]

chef['out_REVENUE'].replace(to_replace = condition_hi,
                            value      = 1,
                            inplace    = True)

# TOTAL MEALS ORDERED
chef['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = chef.loc[0:,'out_TOTAL_MEALS_ORDERED'][chef['TOTAL_MEALS_ORDERED'] 
                                                      > total_meals_ord_hi]
condition_lo = chef.loc[0:,'out_TOTAL_MEALS_ORDERED'][chef['TOTAL_MEALS_ORDERED'] 
                                                      > total_meals_ord_lo]

chef['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                        value      = 1,
                                        inplace    = True)
chef['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_lo,
                                        value      = 1,
                                        inplace    = True)

# UNIQUE MEALS 
chef['out_UNIQUE_MEALS_PURCH']  = 0
condition_hi = chef.loc[0:,'out_UNIQUE_MEALS_PURCH'][chef['UNIQUE_MEALS_PURCH'] 
                                                     > unique_meals_pur_hi]

chef['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                       value      = 1,
                                       inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
chef['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = chef.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][chef['CONTACTS_W_CUSTOMER_SERVICE'] 
                                                              > contact_w_custo_s_hi]
condition_lo = chef.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][chef['CONTACTS_W_CUSTOMER_SERVICE'] 
                                                              > contact_w_custo_s_lo]

chef['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                                value      = 1,
                                                inplace    = True)
chef['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,
                                                value      = 1,
                                                inplace    = True)

# AVG TIME PER SITE VISIT
chef['out_AVG_TIME_PER_SITE_VISIT']  = 0
condition_hi = chef.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][chef['AVG_TIME_PER_SITE_VISIT'] 
                                                          > avg_time_per_site_hi]

chef['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                            value      = 1,
                                            inplace    = True)

# CANCELLATIONS_BEFORE_NOON
chef['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = chef.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][chef['CANCELLATIONS_BEFORE_NOON'] 
                                                            > cancel_bef_noon_hi]

chef['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)

# CANCELLATIONS_AFTER_NOON
chef['out_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = chef.loc[0:,'out_CANCELLATIONS_AFTER_NOON'][chef['CANCELLATIONS_AFTER_NOON'] 
                                                           > cancel_aft_noon_hi]

chef['out_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                             value      = 1,
                                             inplace    = True)

# MOBILE_LOGINS
chef['out_MOBILE_LOGINS'] = 0
condition_hi = chef.loc[0:,'out_MOBILE_LOGINS'][chef['MOBILE_LOGINS'] 
                                                > mobile_log_hi ]
condition_lo = chef.loc[0:,'out_MOBILE_LOGINS'][chef['MOBILE_LOGINS'] 
                                                > mobile_log_lo ]

chef['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                  value      = 1,
                                  inplace    = True)
chef['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                  value      = 1,
                                  inplace    = True)

# PC_LOGINS
chef['out_PC_LOGINS'] = 0
condition_hi = chef.loc[0:,'out_PC_LOGINS'][chef['PC_LOGINS'] 
                                            > pc_log_hi ]

chef['out_PC_LOGINS'].replace(to_replace = condition_hi,
                              value      = 1,
                              inplace    = True)

# WEEKLY_PLAN
chef['out_WEEKLY_PLAN'] = 0
condition_hi = chef.loc[0:,'out_WEEKLY_PLAN'][chef['WEEKLY_PLAN'] 
                                              > weekly_plan_hi]

chef['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# LATE_DELIVERIES
chef['out_LATE_DELIVERIES'] = 0
condition_hi = chef.loc[0:,'out_LATE_DELIVERIES'][chef['LATE_DELIVERIES'] 
                                                  > late_deliv_hi]

chef['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# AVG_PREP_VID_TIME
chef['out_AVG_PREP_VID_TIME'] = 0
condition_hi = chef.loc[0:,'out_AVG_PREP_VID_TIME'][chef['AVG_PREP_VID_TIME'] 
                                                    > avg_prep_vid_time_hi]
condition_lo = chef.loc[0:,'out_AVG_PREP_VID_TIME'][chef['AVG_PREP_VID_TIME'] 
                                                    > avg_prep_vid_time_lo]

chef['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)
chef['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,
                                      value      = 1,
                                      inplace    = True)

# LARGEST_ORDER_SIZE
chef['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = chef.loc[0:,'out_LARGEST_ORDER_SIZE'][chef['LARGEST_ORDER_SIZE'] 
                                                     > largest_order_hi]

chef['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                       value      = 1,
                                       inplace    = True)

# MASTER_CLASSES_ATTENDED
chef['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = chef.loc[0:,'out_MASTER_CLASSES_ATTENDED'][chef['MASTER_CLASSES_ATTENDED'] 
                                                          > master_class_hi]

chef['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                            value      = 1,
                                            inplace    = True)

# AVG_CLICKS_PER_VISIT
chef['out_AVG_CLICKS_PER_VISIT'] = 0
condition_lo = chef.loc[0:,'out_AVG_CLICKS_PER_VISIT'][chef['AVG_CLICKS_PER_VISIT'] 
                                                       < avg_click_per_lo]

chef['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)

# TOTAL_PHOTOS_VIEWED
chef['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = chef.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][chef['TOTAL_PHOTOS_VIEWED'] 
                                                      > total_photo_hi]

chef['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                        value      = 1,
                                        inplace    = True)

# AVG_PRICE_PER_MEAL
chef['out_AVG_PRICE_PER_MEAL'] = 0
condition_hi = chef.loc[0:,'out_AVG_PRICE_PER_MEAL'][chef['AVG_PRICE_PER_MEAL'] 
                                                     > avg_price_per_m_hi]

chef['out_AVG_PRICE_PER_MEAL'].replace(to_replace = condition_hi,
                                       value      = 1,
                                       inplace    = True)


# Developing zero inflation variables where 0 had a big impact
total_photo_viewed_change_at = 0 # zero inflated
weekly_plan_change_at        = 0 # zero inflated



chef['change_TOTAL_PHOTOS_VIEWED'] = 0
condition = chef.loc[0:,'change_TOTAL_PHOTOS_VIEWED'][chef['TOTAL_PHOTOS_VIEWED'] 
                                                      == total_photo_viewed_change_at]

chef['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                           value      = 1,
                                           inplace    = True)

chef['change_WEEKLY_PLAN'] = 0
condition = chef.loc[0:,'change_WEEKLY_PLAN'][chef['WEEKLY_PLAN'] 
                                              == weekly_plan_change_at]

chef['change_WEEKLY_PLAN'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# Dummie variables from the email domain.
# Dataset has to be a DataFrame for .iterrows() to work
chef_email       = pd.DataFrame(chef['EMAIL'])

placeholder_lst  = []

for index, col in chef_email.iterrows():
    split_email  = chef_email.loc[index, 'EMAIL'].split(sep = '@')
    
    placeholder_lst.append(split_email)
    
email_df         = pd.DataFrame(placeholder_lst)
email_df.columns = ['name', 'domain']

# Domain groups
personal_domain     = ['@gmail.com', '@yahoo.com','@protonmail.com']
professional_domain = ['@mmm.com', '@amex.com','@apple.com',
                      '@boeing.com','@caterpillar.com',
                      '@chevron.com','@cisco.com','@cocacola.com',
                      '@disney.com','@dupont.com','@exxon.com',
                      '@ge.org','@goldmansacs.com','@homedepot.com',
                      '@ibm.com','@intel.com','@jnj.com',
                      '@jpmorgan.com','@mcdonalds.com','@merck.com',
                      '@microsoft.com','@nike.com','@pfizer.com',
                      '@pg.com','@travelers.com','@unitedtech.com',
                      '@unitedhealth.com','@verizon.com','@visa.com',
                      '@walmart.com']
junk_domain         = ['@me.com', '@aol.com', '@hotmail.com', '@live.com',
                       '@msn.com','@passport.com']

# For loop categorising the different email domains
placeholder_lst = []

for domain in email_df['domain']:
    
    if '@' + domain in personal_domain:
        placeholder_lst.append('personal')
    elif '@' + domain in professional_domain:
        placeholder_lst.append('professional')
    else:
        placeholder_lst.append('junk')
        
# make the columns into a series to append it to original dataset        
email_df['DOMAIN_GROUP'] = pd.Series(placeholder_lst)

# Add the domain categories column to the original dataset 
chef['DOMAIN'] = email_df['DOMAIN_GROUP']

# Get dummies from the domain variable and drop the original column
one_hot_DOMAIN = pd.get_dummies(chef['DOMAIN'])

# Remove the old and add the 3 new columns
chef           = chef.drop('DOMAIN', axis = 1)
chef           = chef.join([one_hot_DOMAIN])


# declaring set of x-variables
x_variables = ['TOTAL_MEALS_ORDERED','CONTACTS_W_CUSTOMER_SERVICE',
                'EARLY_DELIVERIES','LATE_DELIVERIES', 'AVG_PREP_VID_TIME',
                'FOLLOWED_RECOMMENDATIONS_PCT','LARGEST_ORDER_SIZE',
                'MASTER_CLASSES_ATTENDED', 'MEDIAN_MEAL_RATING',
                'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED',
                'AVG_PRICE_PER_MEAL' ,'out_REVENUE', 'out_TOTAL_MEALS_ORDERED', 
                'out_UNIQUE_MEALS_PURCH','out_AVG_PREP_VID_TIME', 
                'out_AVG_CLICKS_PER_VISIT', 'out_TOTAL_PHOTOS_VIEWED',
                'out_AVG_PRICE_PER_MEAL','junk', 'personal', 'professional']

# Assigning only REVENUE as target variable
target_data      = chef.loc[ : , 'REVENUE']

# All other variables are explanatory
explanatory_data = chef.loc[ : , x_variables]


################################################################################
# Train/Test Split
################################################################################

# The code to divide dataset into a train set (75%) and test set (25%)
# with the random seed number as 222
X_train, X_test, y_train, y_test = train_test_split(
            explanatory_data,
            target_data,
            test_size = 0.25,
            random_state = 222)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# Gradient Boosting Regressor
Gboost      = GradientBoostingRegressor(random_state = 123)

# FITTING the training data
Gboost_fit  = Gboost.fit(X_train, y_train)

# PREDICTING on test data
Gboost_pred = Gboost.predict(X_test)

# saving scoring data for table
Gboost_train_score = Gboost.score(X_train, y_train).round(4)
Gboost_test_score  = Gboost.score(X_test, y_test).round(4)

################################################################################
# Final Model Score (score)
################################################################################

# Final model score on test data
test_score = Gboost.score(X_test, y_test).round(4)


# In[ ]:




