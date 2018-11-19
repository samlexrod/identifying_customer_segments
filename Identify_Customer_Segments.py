#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
from IPython.display import HTML
import re
import os
from helper import parse_datadict

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# for encoding and clustering
from sklearn.preprocessing import LabelBinarizer
from collections import defaultdict

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
def display_html(message, center=False):
    if center:
        center = ';text-align: center'
    else:
        center = ''
    display(HTML('<br/><p style="font-size:20px; text-decoration: underline{0}">{1}<p/>'.format(center, message)))
    
def data_extract(file_name):
    dataframe = pd.read_csv(file_name, delimiter=';')
    print('The dataset {name} has {cols} columns and {rows:,} rows.'.format(name=file_name.split('.')[0], 
                                                                          rows=dataframe.shape[0], 
                                                                          cols=dataframe.shape[1]))
    display(HTML('<br/><p style="font-size:20px; text-decoration: underline">{} Top 5<p/>'.format(file_name.split('.')[0])))
    display(dataframe.head())
    return dataframe


# In[3]:


display_html('test', True)


# In[4]:


# Load in the general demographics data.
azdias = data_extract('Udacity_AZDIAS_Subset.csv')


# In[5]:


# Load in the feature summary file.
feat_info = data_extract('AZDIAS_Feature_Summary.csv')


# In[6]:


azdias.info()


# In[7]:


pd.options.display.max_rows = feat_info.shape[0]
display_html('All AZDIAS_Feature_Summary')
display(feat_info)


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[8]:


try:
    azdias = azdias_original.copy()
except:
    print('First run')
df1, df2, df3, df4 = parse_datadict()


# In[9]:


def nan_parse(dataframe_data, dataframe_parser):
    """
    Parameters
    ----------
    dataframe_data :
    dataframe_parser :
    
    Returns: 
        Transformed np.nan dataframe at index 0
        Converted np.nan count and inpact dataframe at index 1, inpact sorted descending
    """
    
    df_unkmiss = {'ColumnName': [], 'Unk_Missing': [], 'Inpact_Percent_Parsed': []}    
    total_count = dataframe_data.shape[0]

    # go over by each column and parse
    for column in dataframe_data.columns:    

        # getting missing lists per column
        miss_unk = (dataframe_parser.loc[dataframe_parser.attribute == column, 'missing_or_unknown'])
        miss_unk = (miss_unk.item().replace('[', '').replace(']', '').split(','))

        # find the value in missing or unknown list
        flag_val_count = dataframe_data.loc[dataframe_data[column].isin(miss_unk), column].count()

        # fill in the dictionary
        if flag_val_count > 0:
            df_unkmiss['ColumnName'].append(column)
            df_unkmiss['Unk_Missing'].append(flag_val_count)
            df_unkmiss['Inpact_Percent_Parsed'].append(flag_val_count/total_count)

        # convert found missing or unknwon to np.nan
        dataframe_data.loc[dataframe_data[column].isin(miss_unk), column] = np.nan
        
    return dataframe_data, pd.DataFrame(df_unkmiss).sort_values('Inpact_Percent_Parsed', ascending=False)


# In[10]:


def identify_outliers(df_series, mode='non-parametric'):
    mean = df_series.mean() 
    stdev = df_series.std()
    stdev3 = stdev*3
    
    return (mean-stdev3), mean, (mean+stdev3)


# In[11]:


def remove_outliers(data, target_column):    
    return data.loc[data[target_column] < identify_outliers(data[target_column])[2], :]    


# ##### Data Transformation Checkpoint: Parsed Value Keys to Numpy NaN

# In[12]:


azdias_original = azdias.copy()


# In[13]:


azdias, nan_parsed = nan_parse(azdias, feat_info)


# In[14]:


nan_original = pd.DataFrame(azdias_original.isna().sum())
nan_original.reset_index(inplace=True)
nan_original.columns = ['ColumnName', 'Original_Nan']
nan_original = nan_original.loc[nan_original.Original_Nan > 0, :]
nan_original['Inpact_Percent_Ori'] =  nan_original.Original_Nan / azdias_original.shape[0]
nan_original.sort_values('Inpact_Percent_Ori', ascending=False, inplace=True)


# In[15]:


# Identify missing or unknown data values and convert them to NaNs.
nan_all = pd.DataFrame(azdias.isna().sum())
nan_all.reset_index(inplace=True)
nan_all.columns = ['ColumnName', 'Nan_All']
#nan_all = nan_all.loc[nan_all.Nan_All > 0, :]
nan_all['Inpact_Percent_All'] = nan_all.Nan_All / azdias_original.shape[0]
nan_all.sort_values('Inpact_Percent_All', ascending=False, inplace=True)


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[16]:


pd.options.display.float_format = '{:.4f}'.format
# Perform an assessment of how much missing data there is in each column of the
# dataset.
display_html('Original NaN Values Impact - Integrity Check')
display(nan_original.sum())


# In[17]:


display_html('Parsed Unknown or Missing Values - Integrity Check')
display(nan_parsed.sum())


# In[18]:


display_html('All NaN Count, Including Converted - Integrity Check')
display(nan_all.sum())


# In[19]:


nan_merge = pd.merge(nan_all, nan_original, how='left', on='ColumnName')
nan_merge = pd.merge(nan_merge, nan_parsed, how='left', on='ColumnName')
nan_merge.fillna(0, inplace=True)
display_html('Merged NaN Inpact Analysis - Integrity Check ')
display(nan_merge.sum())


# In[20]:


display_html('Merged NaN Inpact Table')
nan_merge.Original_Nan = nan_merge.Original_Nan.map(int)
nan_merge.Unk_Missing = nan_merge.Unk_Missing.map(int)
nan_dtypes = pd.DataFrame(nan_merge.dtypes).T
nan_dtypes.rename({0: 'dtypes'}, axis='index', inplace=True)
display(nan_dtypes, nan_merge)


# In[21]:


display_html('Descriptive Statistics of Original NaN, Unknown or Missing')
nan_descript = nan_merge.loc[nan_merge.Nan_All > 0, :].describe()
nan_descript


# In[22]:


figsize = (16, 10)


# In[23]:


nan_all.hist(figsize=figsize, grid=False, bins=15)
display_html('Outlier Histogram Original Data', True)
plt.show()


# In[24]:


identify_outliers(nan_all.Inpact_Percent_All)


# In[25]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
nan_clean = remove_outliers(nan_all, 'Inpact_Percent_All')
nan_clean.hist(figsize=figsize, grid=False, bins=10)
display_html('Outlier Histogram Reduced Data 1', True)
plt.show()


# In[26]:


identify_outliers(nan_clean.Inpact_Percent_All)


# In[27]:


nan_clean2 = remove_outliers(nan_clean, 'Inpact_Percent_All')
nan_clean2.hist(figsize=figsize, grid=False, bins=8)
display_html('Outlier Histogram Reduced Data 2', True)
plt.show()


# In[28]:


identify_outliers(nan_clean2.Inpact_Percent_All)


# In[29]:


nan_clean3 = remove_outliers(nan_clean2, 'Inpact_Percent_All')
nan_clean3.hist(figsize=figsize, grid=False, bins=8)
display_html('Outlier Histogram Reduced Data Final', True)
plt.show()


# In[30]:


identify_outliers(nan_clean3.Inpact_Percent_All)


# In[31]:


try:
    azdias = azdias_outremoved
except:
    print('Frist Run')


# ##### Data Transformation Checkpoint: Removed Outliers

# In[32]:


azdias_nanparsed = azdias.copy()
azdias = azdias.loc[:, list(nan_clean3.ColumnName)]
azdias.head()


# In[33]:


print('There were %s columns removed.' % (len(azdias_original.columns) - len(azdias.columns)))


# In[34]:


pd.options.display.max_colwidth = 100
df_col_desc = pd.merge(df1, df2, on='key', how='left')

# columns kept
df_col_desc['Kept'] = df_col_desc.ColumnName.apply(lambda x: x in azdias.columns)
display_html('Columns Left for Analysis with Description')
df_col_desc.loc[df_col_desc.Kept == True, ['ColumnName', 'ColNameDescription']].        merge(feat_info.loc[:, ['attribute', 'type']], left_on='ColumnName', right_on='attribute', how='left').        drop('attribute', axis=1)


# In[35]:


display_html('Columns Removed from Analysis with Description')
df_removed_cols = df_col_desc.loc[~df_col_desc.Kept == True, ['key', 'ColumnName', 'ColNameDescription']]
df_removed_cols


# In[36]:


keys = list(df_col_desc.loc[~df_col_desc.Kept == True, ['key', 'ColumnName', 'ColNameDescription']].key)
df3['Kept'] = df3.key.apply(lambda x: x in keys)
display_html('Key Value Maps to Put Removal in Context')
pd.merge(df3.loc[df3.Kept == True, :].iloc[:, :-1], df1, on='key', how='left').merge(df2, on='key', how='left')


# In[37]:


# getting short names
df1['desc_key'] = df1.ColumnName.map(lambda x: x[x.find('_')+1:])
df1.merge(df4.loc[~df4.key.isin(['3.6', '4.2', '4.1', '3.1', '3.5']), :], 
          on='key',
          how='left')\
   .drop('desc_key', axis=1)

# adding missing keys 3.6 and 4.2 to the data_dictionary_value_keys
frames = [df4.loc[df4.key.isin(['3.6', '4.2']), :].          rename(columns={'ColumnShortName':'value_key', 'ColumnMeta':'ValueMaps'}),
          df3.drop('Kept', axis=1)]
data_dict_valkeys = pd.concat(frames).merge(df1, on='key', how='left')
data_dict_valkeys['desc_key'] = data_dict_valkeys.ColumnName                                                 .map(lambda x:                                                       x[x.find('_')+1:])
data_dict_valkeys = data_dict_valkeys.merge(df4                                            .loc[~df4.key                                                     .isin(['3.6', 
                                                            '4.2', 
                                                            '4.1', 
                                                            '3.1', 
                                                            '3.5']), :], 
                                                  on='key',
                                                  how='left')\
                                            .drop('desc_key', axis=1)
data_dict_valkeys.tail()


# In[38]:


#remapping function
def set_remap(dataframe, col_subset):
    
    # dataframe container
    dataframe_return = pd.DataFrame()    
    
    # creating subset
    dataframe = dataframe.loc[:, col_subset]
    
    # remaping keys to valuemaps
    data_dict_subset = ['value_key', 'ValueMaps']      
    
    
    for subset_column in col_subset:
        # changing dtypes to float for numerical values only
        dataframe_dict = data_dict_valkeys.loc[data_dict_valkeys.ColumnName == subset_column, 
                                               data_dict_subset]\
                                          .apply(pd.to_numeric, errors='ignore', downcast='float')
        # merging on value descriptions
        dataframe_merge = pd.merge(dataframe.loc[:, [subset_column]],
                                   dataframe_dict,
                                   left_on=subset_column,
                                   right_on='value_key',
                                   how='left'
                             ).drop(['value_key', subset_column], axis=1).\
                             rename(columns={'ValueMaps': subset_column})    
        
        # creating the dataframe by concatenating individual columns
        dataframe_return = pd.concat([dataframe_return, dataframe_merge], axis=1)
    
    return (dataframe_return)    


# In[39]:


col_subset = ['ALTERSKATEGORIE_GROB', 'AGER_TYP']
ager_typ = set_remap(azdias_nanparsed, col_subset)
ager_typ.loc[~ager_typ.AGER_TYP.isna(), col_subset].groupby(col_subset).size()


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# >Missing data after combining what was already NaN from the raw csv data and the mappings in the information data csv was up to 99.76% and at least 0.32%. Mostly NaN values came from mapping missing or unknwon values to the feat info dataset, which shows the numerical maps of missing or unknown values.
# 
# >Most of the missing data relates to academic titles, birthdates, building types and a colum appearing to describe some elderly personal character typology. However, the data in this column seems to not describe what was intended since people younger than 30 are labeled as elderly. They might have stopped getting this data at some point.
# 
# > The columns droped were 
# ['AGER_TYP',
#  'GEBURTSJAHR',
#  'TITEL_KZ',
#  'ALTER_HH',
#  'KK_KUNDENTYP',
#  'KBA05_BAUMAX']

# In[40]:


try: 
    nan_descript.Unk_Missing = nan_descript.Unk_Missing.map('{:0,.2f}'.format)
    nan_descript.Original_Nan = nan_descript.Original_Nan.map('{:0,.2f}'.format)
    nan_descript.Nan_All = nan_descript.Nan_All.map('{:0,.2f}'.format)
except:pass

nan_descript


# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[41]:


# How much data is missing in each row of the dataset?
total_missing = azdias.isna().sum().sum()
azdias['NaN_Count'] = azdias.isna().sum(axis=1)


print('There are {:,} total missing or unknown fields in the reduced dataset.'.      format(total_missing))
print('Parse validation: We counted a total of {:,} missing or unknwon fields in the NaN_Count column.'.      format(azdias.NaN_Count.sum()))
display_html('Descriptive Statistics of Row NaNs')
azdias.NaN_Count.describe()


# In[42]:


azdias.NaN_Count.hist(figsize=figsize, grid=False, bins=8)
display_html('Row Outlier Histogram', True)
plt.show()


# In[43]:


outlier_factor = 1.5 * (azdias.NaN_Count.quantile(.75) - azdias.NaN_Count.quantile(.25))
median = azdias.NaN_Count.median()
low_bound = median - outlier_factor
high_bound = median + outlier_factor

bad_stand = (azdias.NaN_Count <= low_bound) | (azdias.NaN_Count >= high_bound)
good_stand = ~bad_stand

azdias['Subset_Category'] = None
azdias.loc[bad_stand, 'Subset_Category'] = 'bad standing'
azdias.loc[good_stand, 'Subset_Category'] = 'good standing'
display_html('Number of Rows per <br/>Unique NaN Count and Subset Category')
display(pd.DataFrame(azdias.loc[:, ['NaN_Count', 'Subset_Category']].             groupby(['NaN_Count', 'Subset_Category']).size()).reset_index().rename(columns={0:'Rows'}))


# In[44]:


display_html('Distribution of Separated Sets', center=True)
plt.figure(figsize=(12, 10))
ax = sns.boxplot(y='NaN_Count', x='Subset_Category', data=azdias, palette='Set3', linewidth=1.5)
ax = sns.stripplot(x='Subset_Category', y='NaN_Count', 
              data=azdias.loc[:, ['Subset_Category', 'NaN_Count']], 
              jitter=True, 
              alpha=.01, color='grey', edgecolor='none')
sns.despine()


# In[45]:


display_html('Proportion of Separated Sets', center=True)
plt.figure(figsize=(10,8))
sns.set(style='darkgrid')
ax = sns.countplot(x='Subset_Category', data=azdias)

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:,} - {:.2%}'.format(y, y/azdias.shape[0]), (x.mean(), y), 
            ha='center', va='bottom')


# In[46]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
azdias_good = azdias.loc[azdias.Subset_Category == 'good standing', :].iloc[:, :-2]
azdias_bad = azdias.loc[~(azdias.Subset_Category == 'good standing'), :].iloc[:, :-2]
#target_columns = azdias_good.columns[np.random.randint(0, len(azdias_good.columns), 8)]
target_columns = ['ANREDE_KZ', 'PLZ8_BAUMAX', 'GFK_URLAUBERTYP', 'WOHNLAGE',
       'KBA05_ANTG2', 'RETOURTYP_BK_S', 'FINANZ_ANLEGER',
       'PRAEGENDE_JUGENDJAHRE']


# In[47]:


display_html('Value Keys for Target Columns for Distribution of Values as Reference')
pd.merge(df1.query("ColumnName in {}".format(list(target_columns))), df3, on='key', how='left').    drop(['Kept', 'key'], axis=1)


# In[48]:


fig = plt.figure(figsize=(16, 30))
subplots = [azdias_good.loc[:, target_columns], azdias_bad.loc[:, target_columns]]
plot_id = 1

display_html('Good Set Vs. Bad Set Data Distributions', 20)
for i, column in enumerate(target_columns, 1):    
    for j, subplot in enumerate(subplots):        
        fig.add_subplot(8, 2, plot_id)
        sns.countplot(x = column, data=subplot)        
        plot_id += 1


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# > There were many rows with more than the interquartile range * 1.5. These rows where separated and put in the bad dataset. The distribution of values between the good and bad set was not too different. Robustly, they are quite similar in some aspects. Some examples where data differences differe where in GFK_URLAUBERTYP. Value 5 or Nature fans was significantly higher in the bad set. However, the distribution followed almoust the same shape. In general, I think we can continue with the good set for the analysis as they are robustly similar. 

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[49]:


# How many features are there of each data type?
pd.DataFrame(feat_info.groupby('type').size()).reset_index().rename(columns={0:'col_count'})


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[50]:


frames = [df4.loc[df4.key.isin(['3.6', '4.2']), :].          rename(columns={'ColumnShortName':'value_key', 'ColumnMeta':'ValueMaps'}),
          df3.drop('Kept', axis=1)]
complete_valkeys = pd.concat(frames)


# In[51]:


display_html('Categorical and Mixed Metadata and Values', center=False)
pd.options.display.max_rows = 1000
display(pd.merge(df1, feat_info, left_on='ColumnName', right_on='attribute', how='left')          .drop(['information_level', 'missing_or_unknown', 'attribute'], axis=1)          .query("type == 'categorical' or type == 'mixed'"))


# In[52]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

# assignong booloeans for Droped attributes to avoid pandas warning
feat_info['Droped'] = None
feat_info.loc[feat_info.attribute.isin(df_removed_cols.ColumnName), 'Droped'] = True
feat_info.loc[~feat_info.attribute.isin(df_removed_cols.ColumnName), 'Droped'] = False

# displaying only categorical values
condition = (feat_info.type == 'categorical') & (feat_info.Droped==False)
categorical = list(feat_info.loc[condition, 'attribute'])
azdias_categorical = azdias.loc[:, categorical]
azdias_categorical.head()


# In[53]:


df_categorical_info = pd.concat([azdias_categorical.dtypes, azdias_categorical.nunique()], axis=1).                            rename(columns={0:'data_type', 1:'unique_count'}).sort_values('unique_count')
display_html('Categorical Descriptions')
display(df_categorical_info)


# In[54]:


# anything greater than 2 or object
conditions = (df_categorical_info.data_type == object) | (df_categorical_info.unique_count > 2)
df_encode_columns = df_categorical_info.loc[conditions, :]
display_html('Features to Encode')
display(df_encode_columns)


# In[55]:


print('There are {} categorical variables. '.      format(df_categorical_info.shape[0]) + 
      '\nThere are {} categories with greater than 2 unique or non-numeric values.'.\
      format(df_encode_columns.shape[0]) +
      '\nThere are {} columns in the azdias_categorical dataframe.'.format(azdias_categorical.shape[1]))


# In[56]:


df_encode_columns.index


# In[57]:


# Re-encode categorical variable(s) to be kept in the analysis.

# setting encoder dict for columns
encoder = defaultdict(LabelBinarizer)

# slice only on columns that meet criteria object and unique values greater than 2
azdias_toencode = azdias_categorical.loc[:, df_encode_columns.index]

# kepp original to use later
azdias_toencode_original = azdias_categorical.loc[:, df_encode_columns.index].copy()
azdias_toencode = azdias_toencode.fillna('0').apply(pd.to_numeric, errors='ignore', downcast='integer')

# to put the encoded data
azdias_encoded = pd.DataFrame()

# one big encoder
display_html('Data Encoded:')
for column in azdias_toencode.columns:
    X = encoder[column].fit_transform(azdias_toencode[column].values)
    encoder_colums = [column + '_' + str(enc_value) for enc_value in range(X.shape[1])]
    new_azdias_encoded = pd.DataFrame(X, columns=encoder_colums)
    azdias_encoded = pd.concat([azdias_encoded, new_azdias_encoded], axis=1)   


# In[58]:


display_html('Detemining Nulls that need <br/> to be recoded to 0')
df_toreencode = pd.concat([azdias_categorical.loc[:, df_encode_columns.index].isna().sum(), 
                   azdias_categorical.loc[:, df_encode_columns.index].dtypes], axis=1).\
                        rename(columns={0:'Nulls', 1:'Type'}).\
                        reset_index().\
                        rename(columns={'index':'ColumnName'}).\
                        merge(pd.DataFrame({'encoding': [encoder[column].classes_ \
                                                         for column in df_encode_columns.index]}), 
                              left_index=True, right_index=True)
df_toreencode['encoding_len'] = df_toreencode.encoding.apply(lambda x: len(list(x)))

# creating column label of null values to re-encode
def f(x):
    for label in range(len(x['encoding'])):
        print(x['encoding'])
        if str(label) == '0':
            return x['ColumnName'] + '_' + str(label)
df_toreencode['NullLabel'] = df_toreencode.apply(lambda x: f(x), axis=1)

display(df_toreencode)


# In[59]:


preencode_list = df_toreencode.loc[(df_toreencode.Nulls > 0), :].ColumnName
reencode_list = df_toreencode.loc[(df_toreencode.Nulls > 0), :].NullLabel
reencode_list_not = df_toreencode.loc[~(df_toreencode.Nulls > 0), :].NullLabel


# In[60]:


display_html('Re-encode back to zero')
print(reencode_list)


# In[61]:


display_html('No need to re-encode')
print(reencode_list_not)


# In[62]:


pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[63]:


display_html('Original Data Null Count')
print(azdias_toencode_original.isna().sum())


# In[64]:


# columns that need recoding from 1 to 0 due to being nulls
columns_toreencode = list(df_toreencode.loc[df_toreencode.Nulls > 0, 'ColumnName'])

# for columns in the original dataframe to encode, before endodings
display_html('Re-encoding Encoded Null Values Back to Zero')
for pre_column, column in zip(preencode_list, reencode_list):
    # this will bring all the indexes where there are null values
    indexes = azdias_toencode_original[[pre_column]].loc[azdias_toencode_original                                                    .isna()[pre_column]==True, :]                                                    .index    
    # display to validate null values
    print(column, azdias_encoded.iloc[indexes, :][column].count())
    
    # this will change null labels back to zero using indexes
    i = azdias_encoded.columns.get_loc(column)
    azdias_encoded.iloc[indexes, i] = 0


# #### Validating Endoding Vs. Original Data

# In[65]:


azdias_toencode_original.head().isna().count(axis=1) - azdias_toencode_original.head().isna().sum(axis=1) 


# In[66]:


azdias_encoded.head().sum(axis=1)


# In[67]:


try: azdias = saved_azdias.copy()
except: saved_azdias = azdias.copy()


# In[68]:


print('Azdias has %s columns.' % len(azdias.columns) +      '\nWe are encoding %s columns.' % len(azdias_toencode_original.columns) +      '\nEncoding created %s columns.' % len(azdias_encoded.columns))
azdias.drop(azdias_toencode_original, axis=1, inplace=True)
print('Azdias was reduced to %s columns' % len(azdias.columns))
azdias = pd.concat([azdias, azdias_encoded], axis=1)
print('Azdias total columns are %s' % len(azdias.columns) +      '\n\nAzdias encoding is completed.')


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# > After finding all categorical and mixed variables, I found several of them meeting unique value counts of 2, which does not require encoding. One of those was actually an object type column, thus not numeric. So I flag OST_WEST_KZ to be encoded. The following list needed to be encoded: ['OST_WEST_KZ', 'NATIONALITAET_KZ', 'SHOPPER_TYP', 'LP_STATUS_GROB',
#        'LP_FAMILIE_GROB', 'FINANZTYP', 'ZABEOTYP', 'CJT_GESAMTTYP',
#        'GEBAEUDETYP', 'CAMEO_DEUG_2015', 'LP_STATUS_FEIN', 'LP_FAMILIE_FEIN',
#        'GFK_URLAUBERTYP', 'CAMEO_DEU_2015']
#        
# >I used the LabelBinarizer to encode since it does Label Encoding and Hot Encoding at the same time. But then I encounter errors due to Null values. Since there were not 0 labels left as they where all converted to numpy nan, I filled null values with 0 to be able to run the encoder. Two categorical/mixed colums did not have any nulls: FINANZTYP and ZABEOTYP.
# 
# > After encoding the categorical values. I re-encoded the 0 labels for Null values that where encoded as 1 back to 0 using indexes of the original data. Null values where validated by comparing the sum of null values on the original data vs the encoded data and compared again at a row level using the original data as well.
# 

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[69]:


pd.options.display.max_rows = 1000
data_dict_many = data_dict_valkeys.merge(df2, on='key', how='left').sort_values('ColumnName')
data_dict_many.head()


# In[70]:


data_dict_metalevel = df1.merge(df2, on='key', how='left')                         .merge(df4, left_on='desc_key', 
                                           right_on='ColumnShortName', 
                                           how='left')\
                         .sort_values('ColNameDescription')
data_dict_metalevel.head()


# In[71]:


col_subset = ['ALTERSKATEGORIE_GROB', 'AGER_TYP']
ager_typ = set_remap(azdias_nanparsed, col_subset)
ager_typ.head()


# In[106]:


#remapping function
def set_remap(dataframe, col_subset):
    
    if len(col_subset) >=3:
        dataframe = dataframe.sample(n=3, random_state=42)
    
    def merge_dataframes(dataframe_input, dataframe_dict):
        # merging on value descriptions
        dataframe_merge = pd.merge(dataframe_input.loc[:, [subset_column]],
                                   dataframe_dict,
                                   left_on=subset_column,
                                   right_on='value_key',
                                   how='left'
                             ).drop(['value_key', subset_column], axis=1).\
                             rename(columns={'ValueMaps': subset_column})
        
        if (dataframe_merge[subset_column].count()) == 0:
            return dataframe_input
        else:        
            return dataframe_merge
    
    # dataframe container
    dataframe_return = pd.DataFrame()    
    
    # creating subset
    dataframe = dataframe.loc[:, col_subset]
    
    # remaping keys to valuemaps
    data_dict_subset = ['value_key', 'ValueMaps']      
    
    
    for subset_column in col_subset:
        # changing dtypes to float for numerical values only
        dataframe_dict_original = data_dict_valkeys.loc[data_dict_valkeys                                                        .ColumnName                                                         == subset_column, 
                                                        data_dict_subset]
        
        dataframe_dict = dataframe_dict_original                                .apply(pd.to_numeric, 
                                       errors='ignore', 
                                       downcast='float')
        
        try: dataframe_merge = merge_dataframes(dataframe, dataframe_dict)
        except: dataframe_merge = merge_dataframes(dataframe, dataframe_dict_original)
        
        
        # creating the dataframe by concatenating individual columns
        dataframe_return = pd.concat([dataframe_return, dataframe_merge], axis=1)
        
        #dataframe_return.rename(columns={subset_column: subset_column + '_ORIGINAL'}, inplace=True)
        
    display(dataframe)
    
    return (dataframe_return)    


# In[107]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
pd.options.display.max_rows = 10
col_subset = ['PRAEGENDE_JUGENDJAHRE', 'RELAT_AB']
x = set_remap(azdias, azdias.columns.drop(azdias_encoded.columns))
x


# In[ ]:


data_dict_valkeys.loc[data_dict_valkeys.ColumnName == 'CAMEO_INTL_2015', :]


# In[ ]:


azdias.head(10)


# In[ ]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.

azdias[['CAMEO_INTL_2015']]


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[ ]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)


# In[ ]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[ ]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    
    
    # remove selected columns and rows, ...

    
    # select, re-encode, and engineer column values.

    
    # Return the cleaned dataframe.
    
    


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[ ]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.


# In[ ]:


# Apply feature scaling to the general population demographics data.


# ### Discussion 2.1: Apply Feature Scaling
# 
# (Double-click this cell and replace this text with your own text, reporting your decisions regarding feature scaling.)

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[ ]:


# Apply PCA to the data.


# In[ ]:


# Investigate the variance accounted for by each principal component.


# In[ ]:


# Re-apply PCA to the data while selecting for number of components to retain.


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[ ]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.


# In[ ]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.


# In[ ]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[ ]:


# Over a number of different cluster counts...


    # run k-means clustering on the data and...
    
    
    # compute the average within-cluster distances.
    
    


# In[ ]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.


# In[ ]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.


# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[ ]:


# Load in the customer demographics data.



# In[ ]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[ ]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.


# In[ ]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?


# In[ ]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




