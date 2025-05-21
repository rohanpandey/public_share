#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import re


# In[2]:


base_dir = '/data_218/home1/rohan/0_homelessness'
raw_data_dir = f"{base_dir}/data/raw/"
save_data_dir = f"{base_dir}/data/processed/"


# In[3]:


os.listdir(raw_data_dir)


# ## Initial Analysis

# In[4]:


import os
import pandas as pd
import matplotlib.pyplot as plt

def eda_basic(dataframes: list, dataframes_names: list, visualize=False, verbose=False, save_path=None):
    """ 
    Perform basic exploratory data analysis on a list of pandas DataFrames.
    Args:
    dataframes (list): A list of pandas DataFrames.
    dataframes_names (list, optional): A list of names for the DataFrames. If provided, the names will be used to distinguish between the DataFrames in the results.
    log_path (str, optional): The path to save the log file. If provided, the results will be logged to the file.
    save_path (str, optional): The path to save the csv file. If provided, the results will be saved to the file.
    """
    if not dataframes:
        print("No dataframes provided.")
        return

    if dataframes_names and len(dataframes) != len(dataframes_names):
        print("The number of dataframes and the number of names provided do not match.")
        return

    for df_index, df in enumerate(dataframes):
        name = dataframes_names[df_index] if dataframes_names else f"DataFrame {df_index + 1}"
        print(f"{'*'*40}\n{name}\n{'*'*40}")

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            results = [
                ("First few rows of the dataframe:\n", df.head()),
                ("Columns in the dataframe:\n", df.columns),
                ("Number of rows in the dataframe:\n", df.shape[0]),
                ("Missing values in the dataframe:\n", df.isnull().sum()),
                ("Percentage of missing values in the dataframe:\n", (df.isnull().sum() / df.shape[0]) * 100),
                ("Number of duplicate rows in the dataframe:\n", df.duplicated().sum()),
                ("Data types of columns:\n", df.dtypes),
                ("Number of unique values in each column:\n", df.nunique()),

            ]
            for label, result in results:
                print(f"{label} {result}\n")
                if verbose:
                    print(f"{label}{result}\n\n")
            if verbose:
                print(f"{'X'*40}\n")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            summary_df = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": df.dtypes,
                "Missing Values": df.isnull().sum(),
                "Percentage of Missing Values": (df.isnull().sum() / df.shape[0]) * 100,
                "Number of Unique Values": df.nunique(),
            })
            summary_file_path = save_path+f"{name}_summary.csv"
            summary_df.to_csv(summary_file_path, index=False)
            print(f"Summary saved to {summary_file_path}")
        


# In[5]:


for file in os.listdir(raw_data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(raw_data_dir +file)
        eda_basic([df], [file])


# ## Processing Sociodemographics + Outcomes

# In[6]:


def categorize_bmi(x):
    if x<18.5:
        return 'under'
    if (x>=18.5) & (x<25):
        return 'normal'
    if (x>=25) & (x<30):
        return 'over'
    if x>=30:
        return 'obese'
    else:
        return 'UNK'

def categorize_age(x):
    if (x>=18) & (x<30):
        return '18-29'
    if (x>=30) & (x<40):
        return '30-39'
    if (x>=40) & (x<50):
        return '40-49'
    if (x>=50) & (x<60):
        return '50-59'
    if (x>=60) & (x<70):
        return '60-69'
    if (x>=70) & (x<80):
        return '70-79'
    if (x>=80) & (x<=100):
        return '80-100'
    else:
        return x


# In[7]:


df_2 = df_2 = pd.read_csv(raw_data_dir+"/Roh_homeless2016Cohort_Demog.csv")
df_2 = df_2.drop(columns=['BirthDate','Rlgn','year','avg_height','avg_weight'])

#Replacing NA with N - Because if not available it can be assumed to be not present
df_2['MilitarySexualTraumaFlag'] = df_2['MilitarySexualTraumaFlag'].fillna('N')
df_2['CombatFlag'] = df_2['CombatFlag'].fillna('N')

df_2 = df_2.dropna(subset=['age_on_Jan12016'])
df_2 = df_2[df_2['age_on_Jan12016']>18]
df_2 = df_2[df_2['age_on_Jan12016']<=100]

#Normalizing some values for Gender and Geography
df_2['gender'] = df_2['gender'].replace(['DNA','OTHER'],'UNK').fillna('UNK')
df_2['GISURH'] = df_2['GISURH'].fillna('UNK')
df_2['GISURH'] = df_2['GISURH'].astype(str)

#Normalizing different variations of race
df_2['Race'] = df_2['Race'].replace(['White','WHITE'],'white')
df_2['Race'] = df_2['Race'].replace(['Asian','ASIAN'],'asian')
df_2['Race'] = df_2['Race'].replace(['Native Hawaiian or Other Pacific Islander','NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'],'native_hawaii')
df_2['Race'] = df_2['Race'].replace(['American Indian or Alaska Native','AMERICAN INDIAN OR ALASKA NATIVE'],'american_indian')
df_2['Race'] = df_2['Race'].replace(['WHITE NOT OF HISP ORIG'],'white')
df_2['Race'] = df_2['Race'].replace(['Black or African American','BLACK OR AFRICAN AMERICAN'],'black')
df_2['Race'] = df_2['Race'].replace(['UNKNOWN'],'UNK')

#Normalizing different variations of Ethnicity
df_2['Ethnicity'] = df_2['Ethnicity'].replace(['Hispanic or Latino','HISPANIC OR LATINO'],'hispanic')
df_2['Ethnicity'] = df_2['Ethnicity'].replace(['Not Hispanic or Latino','NOT HISPANIC OR LATINO'],'not_hispanic')
df_2['Ethnicity'] = df_2['Ethnicity'].replace(['UNKNOWN'],'UNK')

df_2['MaritalStatus'] = df_2['MaritalStatus'].astype(str)

#Normalizing some values for state
df_2['STATE'] = df_2['STATE'].replace(['AAAA_MissingUnkOth'],'UNK').fillna('UNK')

#Converting BMI and age into categories
df_2['bmi_category'] = df_2['BMI'].apply(categorize_bmi)
df_2['age_category'] = df_2['age_on_Jan12016'].apply(categorize_age)

df_2 = df_2.drop(columns =[
    'FirstHomelessDate','BMI','age_on_Jan12016'
])
df_2.columns = df_2.columns.str.lower()
df_2.columns = [col if col == 'patienticn' else f"demo_{col}" for col in df_2.columns]


# In[8]:


print(df_2.columns)


# In[9]:


df_4 = pd.read_csv(raw_data_dir+"/homeless_CY2017_Quarterly.csv")
df_4 = df_4.drop(columns=['HOMELESS_2017'])
q1_bool = df_4['HLDate_2017Q1'].notna().astype(int)
q2_bool = df_4['HLDate_2017Q2'].notna().astype(int)
q3_bool = df_4['HLDate_2017Q3'].notna().astype(int)
q4_bool = df_4['HLDate_2017Q4'].notna().astype(int)

df_4['HL_17_Q1'] = q1_bool.astype(int)
df_4['HL_17_Q1_Q2'] = (q1_bool | q2_bool).astype(int)
df_4['HL_17_Q1_Q2_Q3'] = (q1_bool | q2_bool | q3_bool).astype(int)
df_4['HL_17_Q1_Q2_Q3_Q4'] = (q1_bool | q2_bool | q3_bool | q4_bool).astype(int)

print(q1_bool.value_counts())
print(q2_bool.value_counts())
print(q3_bool.value_counts())
print(q4_bool.value_counts())
print(df_4['HL_17_Q1_Q2_Q3_Q4'].value_counts())

df_4 = df_4.drop(columns =[
    'FirstHomelessDate', 'HLDate_2017Q1', 'HLDate_2017Q2', 'HLDate_2017Q3', 'HLDate_2017Q4'
])
df_4.rename(columns = {'PATIENTICN':'patienticn'},inplace = True)
#print(df_4['HOMELESS_2017'].value_counts())
#df_mismatched_values = (df_4[df_4['HOMELESS_2017'] != df_4['HL_17_Q1_Q2_Q3_Q4']])
#print(len(df_mismatched_values))
#df_mismatched_values.to_csv(save_path + '/mismatch_between_values.csv')
#df_4.to_csv(save_path+'/outcomes_2017_quarterly.csv', index = False)


# In[10]:


# Complete code with all helper functions to:
# (1) Aggregate diagnoses by time
# (2) Merge with outcomes and demographics
# (3) Build modeling-ready dataset

import pandas as pd

def aggregate_temporal_diagnoses(
    df,
    df_name,
    patient_id_col='patienticn',
    diag_col='Diagnoses_Rohan',
    date_col='DxDATE',
    level='weekly',
    agg_type='binary'
):
    """
    Aggregates diagnoses over specified temporal intervals.

    Parameters:
        df (pd.DataFrame): Input data with patient ID, diagnosis, and date.
        patient_id_col (str): Patient identifier column.
        diag_col (str): Diagnosis code or name column.
        date_col (str): Date of diagnosis.
        level (str): One of ['daily', 'weekly', 'monthly', 'quarterly'].
        agg_type (str): Aggregation method: 'binary' or 'count'.

    Returns:
        pd.DataFrame: Aggregated dataframe with multi-hot vectors or counts.
    """
    df = df.copy()
    #df.set_index(patient_id_col, inplace = True)
    
    if level == 'yearly':
        df['time_bin'] = df[date_col].dt.year.astype(str)
    
    elif level == 'quarterly':
        year = df[date_col].dt.year.astype(str)
        quarter = df[date_col].dt.quarter.astype(str)
        df['time_bin'] = year + '_Q' + quarter
    
    elif level == 'monthly':
        year = df[date_col].dt.year.astype(str)
        month = df[date_col].dt.month.astype(str).str.zfill(2)
        df['time_bin'] = year + '_M' + month
    
    elif level == 'weekly':
        year = df[date_col].dt.isocalendar().year.astype(str)
        week = df[date_col].dt.isocalendar().week.astype(str).str.zfill(2)
        df['time_bin'] = year + '_W' + week
    
    elif level == 'daily':
        df['time_bin'] = df[date_col].dt.strftime('%Y-%m-%d')
    
    else:
        raise ValueError(f"Unsopported level: {level}")
        
    df = df.dropna(subset=[patient_id_col, diag_col, 'time_bin'])
    #df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    #df = df.dropna(subset=[date_col])
    #df[patient_id_col] = pd.to_numeric(df[patient_id_col], downcast='integer', errors='ignore')
    df['flag'] = 1
    df['pivot_col'] = df_name + '_' + df['time_bin'].astype(str) + '_' + df[diag_col].astype(str)

    #df['pivot_col'] = df['time_bin'].astype(str)+'_'+df[diag_col].astype(str)
    n_patient = df[patient_id_col].nunique()
    n_pivot_cols = df['pivot_col'].nunique()
    #n_time_bins = df['time_bin'].nunique()
    #n_diags = df[diag_col].nunique()
    #expected_cols = n_time_bins * n_diags
    expected_entries = n_patient * n_pivot_cols

    #print("Estimated Total Cols:", expected_cols)
    print("No. Patients: ", n_patient)
    print("No. Pivot_Cols", n_pivot_cols)
    print("Expected Entries:", expected_entries)
    
    try:
        pivot = df.pivot_table(
            index=patient_id_col,
            columns= 'pivot_col',
            values='flag',
            aggfunc='max' if agg_type =='binary' else 'sum',
            fill_value=0
        )
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
            
    n_rows, n_cols = pivot.shape
    total_cells = n_rows * n_cols
    non_zero_cells = (pivot !=0).sum().sum()
    sparsity = 1-(non_zero_cells/total_cells) if total_cells>0 else np.nan
    
    print('Sparsity: ',round(sparsity,4))
    
    pivot = pivot.astype('int32')
    #pivot.columns = [
    #    f"{diag_col}_{time}_{diag.replace(' ', '_')}"
    #    for time, diag in pivot.columns
    #]
    
    return pivot.reset_index()


# ### Processing Diagnoses

# In[11]:


diag_q1_df = pd.read_csv(raw_data_dir+'Dx_ALL_2016Q1_Roh.csv', parse_dates = ['DxDATE'])
diag_q2_df = pd.read_csv(raw_data_dir+'Dx_ALL_2016Q2_Roh.csv', parse_dates = ['DxDATE'])
diag_q3_df = pd.read_csv(raw_data_dir+'Dx_ALL_2016Q3_Roh.csv', parse_dates = ['DxDATE'])
diag_q4_df = pd.read_csv(raw_data_dir+'Dx_ALL_2016Q4_Roh.csv', parse_dates = ['DxDATE'])

diag_df = pd.concat([diag_q1_df,diag_q2_df,diag_q3_df,diag_q4_df],axis=0).reset_index(drop=True)


# In[12]:


drop_values = ['Alcohol Use Disorder', 'Nicotine dependence', 'Opioid Use Disorder', 'Drug Abuse']
diag_df = diag_df[~diag_df['Diagnoses_Rohan'].isin(drop_values)]
diag_df = diag_df.dropna()
diag_df['Diagnoses_Rohan'] = diag_df['Diagnoses_Rohan'].str.lower().str.replace(' ','_')


# In[13]:


diag_list = diag_df['Diagnoses_Rohan'].unique()
print(np.sort(diag_list))


# In[14]:


mental_disorders_list = ['anxiety_disorder', 'bipolar_disorder','dementia', 'depression',
                         'other_neurological_disorders', 'posttraumatic_stress_disorder',
                         'psychoses']
physical_health_diag = diag_df[~diag_df['Diagnoses_Rohan'].isin(mental_disorders_list)]
mental_health_diag = diag_df[diag_df['Diagnoses_Rohan'].isin(mental_disorders_list)]


# In[15]:


print(diag_df['DxDATE'].min())
print(diag_df['DxDATE'].max())


# In[16]:


substance_df = pd.read_csv(raw_data_dir+'SubstanceAbuse_by_visits_2016.csv', parse_dates = ['vizday'])
substance_df.rename(columns={'PatientICN':'patienticn', 'vizday': 'DxDATE', 'SubstanceAbuse_CAT': 'Diagnoses_Rohan'},inplace=True)
substance_df.drop('ICDCode', axis=1, inplace=True)
substance_df.drop_duplicates(inplace=True)
substance_df['Diagnoses_Rohan'] = substance_df['Diagnoses_Rohan'].str.lower().str.replace(' ','_')


# In[17]:


diag_list = substance_df['Diagnoses_Rohan'].unique()
print(np.sort(diag_list))


# In[18]:


print(substance_df['DxDATE'].min())
print(substance_df['DxDATE'].max())


# In[19]:


pain_df = pd.read_csv(raw_data_dir+'Pain_by_visits_2016.csv', parse_dates = ['vizday'])
print(pain_df.shape)
pain_df.rename(columns={'PatientICN':'patienticn', 'vizday': 'DxDATE', 'PAIN_CAT': 'Diagnoses_Rohan'},inplace=True)
pain_df.drop('ICDCode', axis=1, inplace=True)
pain_df['Diagnoses_Rohan'] = 'Pain'
pain_df.drop_duplicates(inplace=True)
print(pain_df.shape)
pain_df = pain_df[pd.to_datetime(pain_df['DxDATE']).dt.year == 2016]
pain_df['Diagnoses_Rohan'] = pain_df['Diagnoses_Rohan'].str.lower().str.replace(' ','_')


sdoh_df = pd.read_csv(raw_data_dir+'SDOH_by_visits_2016.csv', parse_dates = ['vizday'])
sdoh_df.rename(columns={'PatientICN':'patienticn', 'vizday': 'DxDATE', 'SDOH_CAT': 'Diagnoses_Rohan'},inplace=True)
sdoh_df = sdoh_df[pd.to_datetime(sdoh_df['DxDATE']).dt.year == 2016]
sdoh_df['Diagnoses_Rohan'] = sdoh_df['Diagnoses_Rohan'].str.lower().str.replace(' ','_')


# In[20]:


print(pain_df['DxDATE'].min())
print(pain_df['DxDATE'].max())


# In[21]:


print(sdoh_df['DxDATE'].min())
print(sdoh_df['DxDATE'].max())


# In[22]:


print(sdoh_df['Diagnoses_Rohan'].unique())


# In[ ]:


from functools import reduce

df_merged = pd.merge(df_4, df_2, on = 'patienticn', how = 'inner')
agg_level_list = ['yearly','quarterly', 'monthly']#,'weekly']
agg_type_list = ['count','binary']
dataframe_list = [physical_health_diag, mental_health_diag, pain_df, sdoh_df, substance_df]
dataframe_list_names = ['ph_diag', 'mh_diag', 'ph_diag', 'sdoh', 'substance']
for agg_type in agg_type_list:
    for agg_level in agg_level_list:
        pivoted_dfs = []
        for df, df_name in zip(dataframe_list, dataframe_list_names):
            diag_agg_df = aggregate_temporal_diagnoses(df, df_name,level=agg_level,agg_type=agg_type)
            pivoted_dfs.append(diag_agg_df)
        combined = reduce(lambda left,right: pd.merge(left, right, on='patienticn', how ='outer'),pivoted_dfs)
        combined = combined.fillna(0).astype(int)
            
        df_merged_2 = pd.merge(df_merged, combined, on = 'patienticn', how = 'inner')
        df_merged_2.to_csv(f"{save_data_dir}/df_{agg_level}_{agg_type}.csv", index = False)
        print(f"Files saved as {save_data_dir}/df_{agg_level}_{agg_type}.csv")

        def extract_diag_name(col):
            if col == 'patienticn':
                return 'Patient ICN'
            parts = col.split('_')
            # Find the index where the time bin ends
            # Look for patterns like 2016, 2016_Q2, 2016_M01
            for i in range(len(parts)):
                # Year only
                if re.fullmatch(r'\d{4}', parts[i]):
                    # If next part is Q* or M*, include it as part of the time bin
                    if i+1 < len(parts) and (re.fullmatch(r'Q\d+', parts[i+1]) or re.fullmatch(r'M\d+', parts[i+1])):
                        diag_parts = parts[i+2:]
                    else:
                        diag_parts = parts[i+1:]
                    break
            else:
                diag_parts = parts[1:]  # fallback if not found
            diag = ' '.join(' '.join(diag_parts).replace('_', ' ').replace('/', ' or ').split())
            return diag.title()

        col_mapping = {col: extract_diag_name(col) for col in combined.columns}
        mapping_df = pd.DataFrame(list(col_mapping.items()), columns=['column_name', 'readable_name'])
        mapping_df.to_csv(f"{save_data_dir}/df_{agg_level}_{agg_type}_column_mapping.csv", index=False)




