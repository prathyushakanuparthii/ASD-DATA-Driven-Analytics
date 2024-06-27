``` ASD-DATA-Driven-Analytics
Transformed raw video analytics data into understandable metrics and trends using advanced processing methods. Uncovered valuable insights such as viewer engagement patterns and audience demographics
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv(r"C:\Users\DELL\Downloads\data1.csv")


# In[3]:


data.hist() 


# In[3]:


data


# In[4]:


data.corr()


# In[8]:


Q1 = data['probability'].quantile(0.25) 
Q3 = data['probability'].quantile(0.75)
IQR = Q3 - Q1
print("The inter quartile range is:",IQR)
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("lower limit is:",lower_limit)
print("upper limit is:",upper_limit)


# In[9]:


outliers = data[(data['probability'] < lower_limit) | (data['probability'] > upper_limit)]
print("The outliers are:",outliers)


# # business moments

# In[ ]:


data.uid.mean()


# In[ ]:


data.asd_project34_video_id.mean()


# In[ ]:


data.duration.mean()


# In[ ]:


data.probability.mean()


# In[ ]:


data.fps.mean()


# In[ ]:


data.uid.median()


# In[10]:


data.asd_project34_video_id.median()


# In[14]:


data.duration.median()


# In[11]:


data.probability.mean()


# In[12]:


data.fps.median()


# In[13]:


data.uid.mode()


# In[22]:


data.asd_project34_video_id.mode()


# In[23]:


data.duration.mode()


# In[24]:


data.probability.mode()


# In[25]:


data.fps.mode()


# In[26]:


data.uid.std()


# In[27]:


data.asd_project34_video_id.std()


# In[28]:


data.probability.std()


# In[29]:


data.duration.std()


# In[30]:


data.fps.std()


# In[31]:


data.uid.var()


# In[32]:


data.asd_project34_video_id.var()


# In[33]:


data.duration.var()


# In[34]:


data.probability.var()


# In[35]:


data.fps.var()


# In[40]:


a = max(data.uid) - min(data.uid)
a


# In[41]:


a = max(data.asd_project34_video_id) - min(data.asd_project34_video_id)
a


# In[43]:


b = max(data.duration) - min(data.duration)
b


# In[44]:


a = max(data.probability) - min(data.probability)
a


# In[45]:


a = max(data.fps) - min(data.fps)
a


# In[47]:


data.uid.skew()


# In[48]:


data.asd_project34_video_id.skew()


# In[49]:


data.duration.skew()


# In[50]:


data.probability.skew()


# In[51]:


data.fps.skew()


# In[52]:


data.uid.kurt()


# In[53]:


data.asd_project34_video_id.kurt()


# In[54]:


data.duration.kurt()


# In[55]:


data.probability.kurt()


# In[56]:


data.fps.kurt()


# In[57]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[58]:


data.probability.hist()


# In[59]:


sns.kdeplot(data.probability)


# In[60]:


sns.boxplot(data.probability)


# In[61]:


sns.distplot(data.probability)


# In[62]:


sns.displot(data.probability)


# In[65]:


sns.scatterplot(data.probability)


# In[66]:


plt.hist(data.probability)


# In[67]:


import statsmodels.api as sn


# In[69]:


sn.qqplot(data.probability)
plt.show()


# In[72]:


sns.barplot(data.probability)


# In[80]:


df = pd.DataFrame(data)
sns.pairplot(df)


# In[81]:


pip install sweetviz


# In[4]:


import sweetviz as sv


# In[5]:


s = sv.analyze(data)
s.show_html()


# In[85]:


pip install Autoviz


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
from autoviz.AutoViz_Class import AutoViz_Class


# In[7]:


av = AutoViz_Class()


# In[94]:


a = av.AutoViz(r"C:\Users\DELL\Downloads\data1.csv")


# In[95]:


pip install dtale


# In[8]:


import dtale


# In[9]:


d = dtale.show(data)
d.open_browser()


# In[5]:


data.info()


# In[100]:


data.uid = data.uid.astype('float')
data.uid


# In[102]:


a = data.duplicated()
a


# In[103]:


sum(a)


# In[104]:


b = data.drop_duplicates()
b


# In[106]:


Q1 = data['probability'].quantile(0.25)
Q3 = data['probability'].quantile(0.75)
IQR = Q3 - Q1
print("inter quantile range",IQR)


# In[107]:


lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("lower limit:",lower_limit)
print("upper limit as:",upper_limit)


# In[108]:


outliers = data[(data['probability'] > upper_limit) | (data['probability'] < lower_limit)]
outliers


# In[ ]:





# In[118]:


Q1 = data['duration'].quantile(0.25)
Q3 = data['duration'].quantile(0.75)
IQR = Q3 - Q1
print("the inter quantile range is:",IQR)


# In[119]:


lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("lower limit:",lower_limit)
print("upper_limit:",upper_limit)


# In[122]:


outliers = data[(data['duration'] > upper_limit) | (data['duration'] < lower_limit)]
print("the outliers outliers",outliers)


# In[124]:


a = np.where(data['probability'] > upper_limit,True,
             np.where(data['probability'] < lower_limit,True,False))


# In[125]:


trimmed_data = data.iloc[~(a),]
trimmed_data.shape,a.shape


# In[127]:


sns.boxplot(trimmed_data.probability)


# In[128]:


x = np.where(data['probability']>upper_limit,True,np.where(data['probability']<lower_limit,True,False))
data_trim = data.iloc[~(a), ]
data_trim.shape,x.shape


# In[129]:


plt.boxplot(data_trim.probability)


# # winsorization:

# In[132]:


from feature_engine.outliers import Winsorizer


# In[133]:


a = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['probability'])


# In[134]:


x = a.fit_transform(data[['probability']])


# In[135]:


sns.boxplot(x.probability)


# In[136]:


gaussian_method = Winsorizer(capping_method = 'gaussian', tail = 'both', fold = 1.5, variables = ['probability'])
guass_mtds = gaussian_method.fit_transform(data[['probability']])


# In[137]:


sns.boxplot(guass_mtds.probability)


# In[143]:


percentile_method = Winsorizer(capping_method = 'mad', tail = 'both', fold = 0.2, variables =['probability'])
rer = percentile_method.fit_transform(data[['probability']])


# In[142]:


sns.boxplot(rer.probability)


# In[144]:


percentile_method = Winsorizer(capping_method = 'quantiles', tail = 'both', fold = 0.2, variables =['probability'])
rer = percentile_method.fit_transform(data[['probability']])


# In[145]:


sns.boxplot(rer.probability)


# In[147]:


from sklearn.feature_selection import VarianceThreshold

# Assuming 'data' is your DataFrame
# Drop non-numeric columns or encode them numerically before applying VarianceThreshold
numeric_data = data.select_dtypes(include=['number'])

# Set a threshold for variance
threshold = 0.01  # You can adjust this threshold based on your needs

# Apply VarianceThreshold
selector = VarianceThreshold(threshold=threshold)
selector.fit_transform(numeric_data)

# Get the features with non-zero variance
selected_features = numeric_data.columns[selector.get_support()]

# Display the selected features
print("Selected Features:", selected_features)


# # binarization

# In[148]:


data.describe()


# In[153]:


data['new_duration'] = pd.cut(data['duration'],
                             bins = [min(data.duration), data.duration.mean(),max(data.duration)],
                             labels = ['low','high'])


# In[155]:


data['new_duration']
data


# In[156]:


data['new_probability'] = pd.cut(data['probability'],
                                bins = [min(data.probability),data.probability.mean(),max(data.probability)],
                                labels = ['min','max'])


# In[157]:


data['new_probability']
data


# In[159]:


data['new_fps'] = pd.cut(data['fps'],
                        bins = [min(data.fps),data.fps.mean(),max(data.fps)],
                        labels = ['low','high'])
data['new_fps']
data


# # rounding

# In[161]:


data.probability.round()
data


# # one-hot-encoding:

# In[2]:


from sklearn.preprocessing import OneHotEncoder


# In[5]:


x = OneHotEncoder()
a = pd.DataFrame(x.fit_transform(data.iloc[:,2:]).toarray())


# In[6]:


a


# In[12]:


from sklearn.preprocessing import LabelEncoder
a = LabelEncoder()
b = data.iloc[:,2:]
b['fps'] = a.fit_transform(b['fps'])
b['fps']


# In[13]:


b


# In[16]:


x = LabelEncoder()
rer = data.iloc[:,2:]
rer['duration'] = x.fit_transform(rer['duration'])
rer['duration']
rer


# In[3]:


from sklearn.impute import SimpleImputer


# In[6]:


a = SimpleImputer(missing_values = np.nan, strategy = 'mean')
data['fps'] = pd.DataFrame(a.fit_transform(data[['fps']]))
data['fps'].isna().sum()


# In[7]:


a1 = SimpleImputer(missing_values = np.nan, strategy = 'median')
data['uid'] = pd.DataFrame(a1.fit_transform(data[['uid']]))
data['uid'].isna().sum()




# In[10]:


x = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['duration'] = pd.DataFrame(x.fit_transform(data[['duration']]))
data['duration'].isna().sum()


# In[12]:


import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab


# In[16]:


prob = stats.probplot(data.fps,dist = stats.norm,plot=pylab)


# In[17]:


a = stats.probplot(np.log(data.fps),dist = stats.norm , plot = pylab)


# In[20]:


a = np.exp(data.probability)
b = stats.probplot(data.probability, dist = stats.norm, plot = pylab)


# In[22]:


a1 = stats.probplot(np.log(data.probability), dist = stats.norm, plot = pylab)


# In[24]:


a_data, a_lambda= stats.boxcox(data.probability)
a_data, a_lambda


# In[34]:


fig, ax = plt.subplots(1,2)


# In[28]:


fitted_data,fitted_lambda = stats.boxcox(data.uid)
fitted_data, fitted_lambda


# In[35]:


fig, ax = plt.subplots(1,2)


# # featurescaling/ feature shrinking:

# In[36]:


from sklearn.preprocessing import MinMaxScaler


# In[37]:


a = MinMaxScaler()


# In[51]:


a1 = data.describe()
a1


# In[54]:


x1 = a.fit_transform(a1)
z = pd.DataFrame(x1)
z


# In[56]:


c = z.describe()
c


# In[58]:


from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
m = data.describe()
m


# In[62]:


scaling_method = standard.fit_transform(m)
converting_dataframe = pd.DataFrame(scaling_method)
a = converting_dataframe.describe()
a


# In[63]:


from sklearn.preprocessing import RobustScaler
a = RobustScaler()
b = data.describe()
b


# In[64]:


trans = a.fit_transform(b)
res = pd.DataFrame(trans)
last = res.describe()
last


# In[16]:


a = data.duplicated()
a


# In[17]:


sum(a)


# In[4]:


data.mean()


# In[5]:


data.var()


# In[6]:


data.kurt()


# In[8]:


Q1 = data['probability'].quantile(0.25)
Q3 = data['probability'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5*(IQR)
upper_limit = Q3 + 1.5*(IQR)
print("lower_limit",lower_limit)
print("lower_limit",upper_limit)
print(IQR)
```
