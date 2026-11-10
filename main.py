#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pandas as pd
import numpy as np
from utils import create_samples , models_comparison_metrics , train_test_models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from itertools import combinations_with_replacement

print("Libraries loaded")


# In[ ]:


pd.set_option('display.max_columns', None)


# In[155]:


df = pd.read_csv("loan.csv" , 
                 usecols = ['loan_status' , 'annual_inc' , 'loan_amnt' , 'int_rate' , 'installment' , 'issue_d'] ,
                 nrows = 800_000
                )
df.shape


# In[156]:


df = df[df['loan_status'].isin(['Fully Paid' , 'Charged Off'])]


# In[157]:


df['y'] = (df['loan_status'] == 'Fully Paid').astype(int)


# In[158]:


df.shape


# In[159]:


df = df.dropna()


# In[160]:


test_size = 100000

df_train = df.iloc[test_size : ]
df_test = df.iloc[ : test_size]


# In[161]:


features = ['annual_inc' , 'loan_amnt' , 'int_rate' , 'installment']
target = 'y'

X_test = df_test[features]
y_test = df_test[target]


# In[162]:


SEEDS = range(5)


# In[163]:


models = [
    ("logistic" , LogisticRegression(max_iter = 2000)) ,
    ("tree" , DecisionTreeClassifier(max_depth = 5)) ,
    ("forest" , RandomForestClassifier(n_estimators = 100 , max_depth = 5))
]

train_sizes = [500 , 1000 , 3000 , 10000 , 70000]
data_overlaps = [0 , 0.25 , 0.5 , 0.75 , 1]


# In[164]:


results = []
counter = 0

for (m1_name , m1) , (m2_name , m2) in combinations_with_replacement(models , 2):
    for sample_size in train_sizes:
        for data_overlap in data_overlaps:
            for exp_number, seed in enumerate(SEEDS, start=1):
                counter += 1
                if counter % 25 == 0:
                    print(f"Completed {counter}/750")
                
                model1 = clone(m1).set_params(random_state = seed)
                model2 = clone(m2).set_params(random_state = seed)

                X_train_1 , y_train_1 , X_train_2 , y_train_2 , obv_MAD , obv_MAD_to_average = create_samples(
                    df_train , sample_size , data_overlap , seed)

                y_pred_1 , y_prob_1 , y_pred_2 , y_prob_2 = train_test_models(
                    model1 , model2 , X_train_1 , y_train_1 , X_train_2 , y_train_2 , X_test)

                exp_results = models_comparison_metrics(
                    m1_name , m2_name , y_pred_1 , y_prob_1 , y_pred_2 , y_prob_2 , y_test , sample_size , data_overlap , exp_number , obv_MAD , obv_MAD_to_average , seed)

                
                results.append(exp_results)




results_df = pd.DataFrame(results)
print("Experiment finished, results are loaded")


# In[177]:


results_df.to_csv("results.csv", index=False)


# In[173]:


group_cols = ["model1", "model2", "sample_size", "data_overlap"]

metric_cols = [
    "pred_corr",
    "prob_corr",
    "P_1_1",
    "P_1_0",
    "P_0_1",
    "P_0_0",
    "agreement_rate",
    "jaccard",
    "err_corr",
    "m1_fn_rate",
    "m1_fp_rate",
    "m2_fn_rate",
    "m2_fp_rate",
    "MAD",
    "MAD_to_average"
]

results_aggregate = (
    results_df
    .groupby(group_cols)[metric_cols].mean()
    .reset_index()
)


# In[178]:


results_aggregate.to_csv("results_aggregate.csv", index=False)


# In[ ]:




