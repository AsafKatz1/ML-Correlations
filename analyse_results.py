#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


print("Libraries loaded")


# In[2]:


results_aggregate = pd.read_csv("results_aggregate.csv")
results_aggregate.shape


# In[18]:


fig , ax = plt.subplots(1, 2 , figsize=(18,6))

fig.suptitle("All Data", fontsize=22)


sns.lineplot(
    data=results_aggregate,
    x="data_overlap",
    y="prob_corr",
    hue="sample_size",
    palette="tab10" ,
    ax = ax[0]
)
ax[0].set_xlabel("Data Overlap", fontsize=16)
ax[0].set_ylabel("Probability (Score) Correlation", fontsize=16)



sns.lineplot(
    data=results_aggregate,
    x="data_overlap",
    y="pred_corr",
    hue="sample_size",
    palette="tab10" ,
    ax = ax[1]
)
ax[1].set_xlabel("Data Overlap", fontsize=16)
ax[1].set_ylabel("Prediction Correlation", fontsize=16)

plt.show()


# In[19]:


fig , ax = plt.subplots(1, 2 , figsize=(18,6))

fig.suptitle("Different Models (Model1 != Model2)", fontsize=22)


sns.lineplot(
    data=results_aggregate[results_aggregate['model1'] != results_aggregate['model2']],
    x="data_overlap",
    y="prob_corr",
    hue="sample_size",
    palette="tab10" ,
    ax = ax[0]
)
ax[0].set_xlabel("Data Overlap", fontsize=16)
ax[0].set_ylabel("Probability (Score) Correlation", fontsize=16)



sns.lineplot(
    data=results_aggregate[results_aggregate['model1'] != results_aggregate['model2']],
    x="data_overlap",
    y="pred_corr",
    hue="sample_size",
    palette="tab10" ,
    ax = ax[1]
)
ax[1].set_xlabel("Data Overlap", fontsize=16)
ax[1].set_ylabel("Prediction Correlation", fontsize=16)


plt.show()


# In[20]:


fig , ax = plt.subplots(1, 2 , figsize=(18,6))

fig.suptitle("Same Model (Model1 = Model2)", fontsize=22)


sns.lineplot(
    data=results_aggregate[results_aggregate['model1'] == results_aggregate['model2']],
    x="data_overlap",
    y="prob_corr",
    hue="sample_size",
    palette="tab10" ,
    ax = ax[0]
)
ax[0].set_xlabel("Data Overlap", fontsize=16)
ax[0].set_ylabel("Probability (Score) Correlation", fontsize=16)



sns.lineplot(
    data=results_aggregate[results_aggregate['model1'] == results_aggregate['model2']],
    x="data_overlap",
    y="pred_corr",
    hue="sample_size",
    palette="tab10" ,
    ax = ax[1]
)
ax[1].set_xlabel("Data Overlap", fontsize=16)
ax[1].set_ylabel("Prediction Correlation", fontsize=16)


plt.show()


# In[ ]:




