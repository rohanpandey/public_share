import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

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
        if visualize:
            if save_path:
                msno.matrix(df, figsize=(10, 6), savefig=save_path + f'/{name}_missing_values.png')
            else:
                msno.matrix(df)
            plt.show()
        

def add_first_visit_label(filtered_admissions_df, patient_column='subject_id', admit_time='admittime', label_column='first_visit_label'):
    """
    Add a label to indicate the first visit for each patient.
    Args:
    filtered_admissions_df (pd.DataFrame): The Admissions Table DataFrame (filtered as per requirements).
    patient_column (str, optional): The column name for the patient ID in the DataFrame.
    admit_time (str, optional): The column name for the admission time in the DataFrame.
    Returns:
    pd.DataFrame: The admissions DataFrame with an added 'first_diagnosis_label' column.
    """
    admissions_sorted = filtered_admissions_df.copy()
    admissions_sorted = admissions_sorted.sort_values(by=[patient_column, admit_time])
    admissions_sorted[label_column] = admissions_sorted.groupby(patient_column).cumcount().eq(0).astype(int)
    return admissions_sorted

def add_first_diagnosis_label(admissions_df, filtered_diagnosis_df, admission_column='hadm_id', patient_column='subject_id', admit_time='admittime', label_column='first_diagnosis_label'):
    """
    Add a label to indicate the first diagnosis for each patient.
    Args:
    admissions_df (pd.DataFrame): The Admissions Table DataFrame.
    filtered_diagnosis_df (pd.DataFrame): The Diagnosis Table DataFrame (filtered as per requirements).
    admission_column (str, optional): The column name for the admission ID in the DataFrame.
    patient_column (str, optional): The column name for the patient ID in the DataFrame.
    admit_time (str, optional): The column name for the admission time in the DataFrame.
    Returns:
    pd.DataFrame: The admissions DataFrame with an added 'first_diagnosis_label', 'first_diagnosis_admittime' and 'before_first_diagnosis' columns.    
    """
    filtered_hadm_ids = filtered_diagnosis_df[admission_column].unique()
    filtered_subject_ids = filtered_diagnosis_df[patient_column].unique()   

    filtered_admissions_hadm_ids = admissions_df[admissions_df[admission_column].isin(filtered_hadm_ids)].copy()
    filtered_admissions_hadm_ids = add_first_visit_label(filtered_admissions_hadm_ids, patient_column=patient_column, label_column=label_column)
    filtered_admissions_hadm_ids['admittime'] = pd.to_datetime(filtered_admissions_hadm_ids[admit_time])
    first_diagnosis_admittime = filtered_admissions_hadm_ids[filtered_admissions_hadm_ids[label_column] == 1].groupby(patient_column)[admit_time].min().reset_index()
    first_diagnosis_admittime.columns = [patient_column, 'first_diagnosis_admittime']

    filtered_admissions_subject_ids = admissions_df[admissions_df[patient_column].isin(filtered_subject_ids)].copy()
    filtered_admissions = filtered_admissions_subject_ids.merge(first_diagnosis_admittime, on=patient_column, how='left')
    filtered_admissions['before_first_diagnosis'] = filtered_admissions[admit_time] < filtered_admissions['first_diagnosis_admittime']
    filtered_admissions['after_first_diagnosis'] = filtered_admissions[admit_time] > filtered_admissions['first_diagnosis_admittime']
    filtered_admissions['first_diagnosis_label'] = filtered_admissions['first_diagnosis_admittime'].eq(filtered_admissions[admit_time]).astype(int)
    return filtered_admissions

#Question 1: How many patients have a certain diagnosis? (Selected with ICD coding) 
def get_filtered_diagnoses_df(diagnosis_df, filtered_diagnoses_code_list=['Z590'], code_column='icd_code', patient_column='subject_id', description_name='Homelessness', visualise=False, verbose=False, save_path=None):
    """
    Filter the Diagnoses_ICD Table based on the selected diagnosis codes and return the filtered DataFrame.
    Args:
    diagnosis_df (pd.DataFrame): The Diagnoses_ICD Table DataFrame.
    filtered_diagnoses_code_list (list, optional): The list of diagnosis codes to filter the DataFrame.
    code_column (str, optional): The column name for the diagnosis code in the DataFrame.
    patient_column (str, optional): The column name for the patient ID in the DataFrame.
    description_name (str, optional): The description of the selected diagnosis.
    visualise (bool, optional): Whether to visualise the occurrences of the selected diagnosis.
    verbose (bool, optional): Whether to print the occurrences of the selected diagnosis.
    save_path_data (str, optional): The path to save the visualisation.
    Returns:
    pd.DataFrame: The filtered Diagnoses_ICD Table DataFrame.
    """

    filtered_diagnosis_df = diagnosis_df[diagnosis_df[code_column].isin(filtered_diagnoses_code_list)].copy()
    if filtered_diagnosis_df.empty:
        print(f"No occurrences of {description_name} in the Diagnoses_ICD Table.")
        return filtered_diagnosis_df

    if verbose:
        for code in filtered_diagnoses_code_list:
            code_df = diagnosis_df[diagnosis_df[code_column] == code]
            print(f"Occurrences of {code} in the Diagnoses_ICD Table: {code_df.shape[0]}")
            print(f"Number of patients with diagnosis of {code}: {code_df[patient_column].nunique()}")

    if visualise:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=code_column, data=filtered_diagnosis_df)
        plt.title(f'Diagnoses_ICD Table: Occurrences of {description_name}')
        if save_path:
            plt.savefig(f'{save_path}/{description_name}_diagnosis_occurrences.png')
        plt.show()
    return filtered_diagnosis_df

#Question 2: How many visits do patients with a certain diagnosis have? 
def plot_patient_visit_counts(admissions_df, filtered_diagnosis_df, admission_column='hadm_id', patient_column='subject_id', admit_time='admittime', label_column='first_diagnosis_label', description_name='homelessness', visualise=False, verbose=False, save_path=None):
    """
    Get the distribution of visit count for each patient with a certain diagnosis.
    Args:
    filtered_diagnosis_df (pd.DataFrame): The filtered Diagnoses_ICD Table DataFrame with the selected diagnosis.
    admissions_df (pd.DataFrame): The Admissions Table DataFrame.
    patient_column (str, optional): The column name for the patient ID in the DataFrame.
    admit_time (str, optional): The column name for the admission time in the DataFrame.
    label_column (str, optional): The column name for the label in the DataFrame.
    description_name (str, optional): The description of the selected diagnosis.
    admission_column (str, optional): The column name for the admission ID in the DataFrame.
    visualise (bool, optional): Whether to visualise the distribution of visit count.
    verbose (bool, optional): Whether to print the distribution of visit count.
    Returns:
    Plot: The plot of the distribution of visit count.
    """
    filtered_admissions = add_first_diagnosis_label(admissions_df, filtered_diagnosis_df, admission_column=admission_column, patient_column=patient_column, admit_time=admit_time, label_column=label_column)
    
    def print_visit_counts(filtered_admissions, description_name):
        print("For " + description_name)
        print("Number of unique patients with more than one visit:", filtered_admissions[filtered_admissions['before_first_diagnosis']][patient_column].nunique())
        print("Number of total visits which are before first diagnosis:", filtered_admissions[filtered_admissions['before_first_diagnosis']].shape[0])
        print("Number of total visits which are first diagnosis:", filtered_admissions[filtered_admissions['first_diagnosis_label'] == 1].shape[0])
        print("Number of total visits which are after first diagnosis:", filtered_admissions[filtered_admissions['after_first_diagnosis']].shape[0])
        print("Number of total visits:", filtered_admissions.shape[0])
    
    def plot_visit_counts(filtered_admissions, description_name, save_path):
        def plot_bar(data, title, xlabel, ylabel, filename):
            plt.figure(figsize=(10, 6))
            sns.barplot(x=data.index, y=data.values, color='skyblue')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=90)
            if save_path:
                plt.savefig(save_path + f'/{filename}.png')
            plt.show()
        
        before_first_diagnosis_visit_counts = filtered_admissions[filtered_admissions['before_first_diagnosis']].groupby(patient_column).size().value_counts().sort_index()
        plot_bar(before_first_diagnosis_visit_counts, 'Number of visits before first diagnosis of ' + description_name, 'Visit Counts', 'Number of Patients', description_name + '_visit_counts_before_first_diagnosis')
        plot_bar(before_first_diagnosis_visit_counts.cumsum(), 'Number of visits before first diagnosis (cumulative) of ' + description_name, 'Visit Counts', 'Cumulative Number of Patients', description_name + '_visit_counts_cumulative_before_first_diagnosis')
        
        after_first_diagnosis_counts = filtered_admissions[filtered_admissions['after_first_diagnosis']].groupby(patient_column).size().value_counts().sort_index()
        plot_bar(after_first_diagnosis_counts, 'Number of visits after first diagnosis of ' + description_name, 'Visit Counts', 'Number of Patients', description_name + '_visit_counts_after_first_diagnosis')
        plot_bar(after_first_diagnosis_counts.cumsum(), 'Number of visits after first diagnosis (cumulative) of ' + description_name, 'Visit Counts', 'Cumulative Number of Patients', description_name + '_visit_counts_cumulative_after_first_diagnosis')
    
    if verbose:
        print_visit_counts(filtered_admissions, description_name)
    
    if visualise:
        plot_visit_counts(filtered_admissions, description_name, save_path)

def get_distribution_numerical(filtered_diagnosis_df, df, column, verbose=True, min_value=0, max_value=120, bins=10, description=""):
    filtered_df = df[df['subject_id'].isin(filtered_diagnosis_df['subject_id'])]
    distribution = filtered_df[column].describe()
    col_distribution = filtered_df[column].value_counts(dropna=False, bins=range(min_value, max_value, bins)).sort_index()
    col_distribution_normalized = filtered_df[column].value_counts(dropna=False, bins=range(min_value, max_value, bins), normalize=True).sort_index()*100
    if verbose:
        print(f"{description} Distribution of Patients:")
        print(distribution)
        print(f"Percentage {description.lower()} distribution of patients:")
        print(col_distribution_normalized)
        print(f"Percentage {description.lower()} distribution of patients:")
        print(col_distribution_normalized)
    return distribution, col_distribution, col_distribution_normalized

def get_distribution_categorical(filtered_diagnosis_df, df, column, verbose=True, description=""):
    filtered_df = df[df['subject_id'].isin(filtered_diagnosis_df['subject_id'])]
    distribution = filtered_df[column].value_counts(dropna=False)
    distribution_normalized = distribution / distribution.sum() * 100
    
    if verbose:
        print(f"{description} Distribution of Patients:")
        print(distribution)
        print(f"Percentage {description.lower()} distribution of patients:")
        print(distribution_normalized)
    
    return distribution, distribution_normalized

# Question 3.a: What is the age distribution of patients with a certain diagnosis at the time of first diagnosis?
def get_age_distribution(filtered_admissions_df, patients_df, age_column = 'anchor_age', patient_column = 'subject_id', verbose = True, visualise = False):
    return get_distribution_numerical(filtered_admissions_df, patients_df, age_column, verbose, min_value=0, max_value=120, bins=10, description="Age")

def get_gender_distribution(filtered_diagnosis_df, patients_df, verbose=True):
    return get_distribution_categorical(filtered_diagnosis_df, patients_df, 'gender', verbose, "Gender")

def get_marital_distribution(filtered_diagnosis_df, admissions_df, verbose=True):
    return get_distribution_categorical(filtered_diagnosis_df, admissions_df, 'marital_status', verbose, "Marital Status")

def get_race_distribution(filtered_diagnosis_df, admissions_df, verbose=True):
    return get_distribution_categorical(filtered_diagnosis_df, admissions_df, 'race', verbose, "Race")

# Question 3: What is the sociodemographic distribution of patients with a certain diagnosis?
def get_sociodemographic_distribution(admissions_df, patients_df, filtered_diagnosis_df, save_path=None):    
    #IMP Note: Filter the data to only the patients that we will use for our final problems - because we might want to do this only for patients that have more than 1 visit and so on
    #TODO: Add this filter later
    gender_distribution, gender_distribution_normalized = get_gender_distribution(filtered_diagnosis_df, patients_df, verbose=False)
    age_desc, age_distribution, age_distribution_normalized = get_age_distribution(filtered_diagnosis_df, patients_df, verbose=False)
    marital_distribution, marital_distribution_normalized = get_marital_distribution(filtered_diagnosis_df, admissions_df, verbose=False)
    race_distribution, race_distribution_normalized = get_race_distribution(filtered_diagnosis_df, admissions_df, verbose=False)

    data = {
        'Sociodemographic Category': ['Gender', 'Age', 'Marital Status', 'Race'],
        'Class': [gender_distribution.index.tolist(), age_distribution.index.tolist(), marital_distribution.index.tolist(), race_distribution.index.tolist()],
        'Absolute Values': [gender_distribution.values.tolist(), age_distribution.values.tolist(), marital_distribution.values.tolist(), race_distribution.values.tolist()],
        'Percentage Values': [gender_distribution_normalized.values.tolist(), age_distribution_normalized.values.tolist(), marital_distribution_normalized.values.tolist(), race_distribution_normalized.values.tolist()],
    }
    
    df = pd.DataFrame(data)
    dfs = []
    for index, row in df.iterrows():
        feature = row['Sociodemographic Category']
        classes = row['Class']
        absolute = row['Absolute Values']
        percentages = row['Percentage Values']
        
        temp_df = pd.DataFrame({
            'Feature': [feature] + [''] * (len(classes) - 1),  # Include feature name only in the first row
            'Class': classes,
            'Absolute': absolute,
            'Percentages': percentages
        })
        dfs.append(temp_df)

    result_df = pd.concat(dfs, ignore_index=True)
    result_df.to_csv(save_path + '/sociodemographic_distribution.csv', index=False)

#Continue from here
# Question 4: What is the distribution of duration of stay for each visit?
def get_length_of_stay(admissions_df, filtered_diagnosis_df, visualise=False, verbose=True, save_path=None):
    def calculate_length_of_stay(df):
        df['length_of_stay'] = (pd.to_datetime(df['dischtime']) - pd.to_datetime(df['admittime'])).dt.days
        return df

    def plot_length_of_stay(df, title, filename):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['length_of_stay'], kde=True)
        plt.title(title)
        plt.xlabel('Length of Stay')
        plt.ylabel('Count')
        plt.show()
        if filename:
            plt.savefig(filename)

    def plot_cumulative_length_of_stay(df, title, filename):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['length_of_stay'], kde=True, cumulative=True)
        plt.title(title)
        plt.xlabel('Length of Stay')
        plt.ylabel('Count')
        plt.show()
        if filename:
            plt.savefig(filename)

    filtered_admissions_df = calculate_length_of_stay(admissions_df[admissions_df['hadm_id'].isin(filtered_diagnosis_df['hadm_id'])])

    if visualise:
        plot_length_of_stay(filtered_admissions_df, 'Length of Stay Distribution of Patients with Diagnosis of Homelessness (Homelessness Admissions)', save_path + '/length_of_stay.png' if save_path else None)
        plot_cumulative_length_of_stay(filtered_admissions_df, 'Length of Stay Distribution of Patients with Diagnosis of Homelessness (Homelessness Admissions) (Cumulative)', save_path + '/length_of_stay_cumulative.png' if save_path else None)

    if verbose:
        print("Length of Stay Distribution of Patients with Diagnosis of Homelessness (Homelessness Admissions):")
        print(filtered_admissions_df['length_of_stay'].value_counts().describe())

    filtered_admissions_df = calculate_length_of_stay(add_first_diagnosis_label(admissions_df, filtered_diagnosis_df))
    filtered_admissions_df = filtered_admissions_df[filtered_admissions_df['before_first_diagnosis']]

    if visualise:
        plot_length_of_stay(filtered_admissions_df, 'Length of Stay Distribution of Patients with Diagnosis of Homelessness (Admissions before First Diagnosis)', save_path + '/length_of_stay_before.png' if save_path else None)
        plot_cumulative_length_of_stay(filtered_admissions_df, 'Length of Stay Distribution of Patients with Diagnosis of Homelessness (Admissions before First Diagnosis) (Cumulative)', save_path + '/length_of_stay_before_cumulative.png' if save_path else None)

    if verbose:
        print("Length of Stay Distribution of Patients with Diagnosis of Homelessness (Admissions before First Diagnosis):")
        print(filtered_admissions_df['length_of_stay'].value_counts().describe())


# Question 6: What is the count of different services for patients with a certain diagnosis?
def get_service_distribution(admissions_df, filtered_diagnosis_df, services_df, description_name='Homelessness', visualise=True, verbose=True, save_path=None):
    def plot_and_save(filtered_services_df, title, filename):
        if visualise:
            plt.figure(figsize=(10, 6))
            sns.histplot(filtered_services_df['curr_service'])
            plt.title(title)
            plt.xlabel('Service')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()      
            plt.show()
            plt.savefig(filename)
        if verbose:
            print(filtered_services_df['curr_service'].value_counts())

    # First diagnosis
    filtered_admissions_df = add_first_diagnosis_label(admissions_df, filtered_diagnosis_df)
    filtered_admissions_df = filtered_admissions_df[filtered_admissions_df['first_diagnosis_label'] == 1]
    filtered_services_df = services_df[services_df['hadm_id'].isin(filtered_admissions_df['hadm_id'])]
    plot_and_save(filtered_services_df, 
                  f'Services Distribution of Patients with Diagnosis of {description_name} at the Time of First Diagnosis (Selected Diagnoses)', 
                  save_path + '/service_distribution.png' if save_path else None)

    # All selected diagnoses
    filtered_admissions_df = add_first_diagnosis_label(admissions_df, filtered_diagnosis_df)
    filtered_services_df = services_df[services_df['hadm_id'].isin(filtered_admissions_df['hadm_id'])]
    plot_and_save(filtered_services_df, 
                  f'Services Distribution of Patients with Diagnosis of {description_name} for all admissions (Selected Diagnoses)', 
                  save_path + '/service_distribution_all.png' if save_path else None)

    # Before first diagnosis    
    filtered_admissions_df = add_first_diagnosis_label(admissions_df, filtered_diagnosis_df)
    filtered_admissions_df = filtered_admissions_df[filtered_admissions_df['before_first_diagnosis']]
    filtered_services_df = services_df[services_df['hadm_id'].isin(filtered_admissions_df['hadm_id'])]
    plot_and_save(filtered_services_df, 
                  f'Services Distribution of Patients with Diagnosis of {description_name} before First Diagnosis (All Admissions)', 
                  save_path + '/service_distribution_before.png' if save_path else None)
    
# Question 7: What is the count of different diagnoses for patients with a certain diagnosis?
# Add a log save for the prints if any
def get_diagnosis_distribution(admissions_df, diagnosis_df, filtered_diagnosis_df, visualise=False, verbose=True, save_path=None):
    def plot_and_save(filtered_diagnosis_df, title, filename):
        if visualise:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=filtered_diagnosis_df, x='icd_code', discrete=True)
            plt.title(title)
            plt.xlabel('ICD Code')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            plt.savefig(filename)
        if verbose:
            print(filtered_diagnosis_df['icd_code'].value_counts())

    #Plot and save the top 20 diagnoses at the time of first diagnosis
    filtered_admissions_df = add_first_diagnosis_label(admissions_df, filtered_diagnosis_df)
    filtered_admissions_df = filtered_admissions_df[filtered_admissions_df['first_diagnosis_label'] == 1]
    filtered_diagnosis_df = diagnosis_df[diagnosis_df['hadm_id'].isin(filtered_admissions_df['hadm_id'])]
    diagnosis_counts = filtered_diagnosis_df['icd_code'].value_counts()
    top_diagnoses_list = diagnosis_counts.head(20).keys().tolist()
    filtered_diagnosis_df = diagnosis_df[diagnosis_df['icd_code'].isin(top_diagnoses_list)]
    filtered_diagnosis_df = filtered_diagnosis_df.sort_values(by='icd_code')
    plot_and_save(filtered_diagnosis_df, "Top 20 Diagnoses at the time of first diagnosis", save_path + '/diagnosis_distribution_top_20.png' if save_path else None)

    #Plot and save the top 20 diagnoses for all admissions of patients with the diagnosis
    filtered_diagnosis_df = diagnosis_df[diagnosis_df['hadm_id'].isin(filtered_diagnosis_df['hadm_id'])]
    diagnosis_counts = filtered_diagnosis_df['icd_code'].value_counts()
    top_diagnoses_list = diagnosis_counts.head(20).keys().tolist()
    filtered_diagnosis_df = diagnosis_df[diagnosis_df['icd_code'].isin(top_diagnoses_list)]
    filtered_diagnosis_df = filtered_diagnosis_df.sort_values(by='icd_code')
    plot_and_save(filtered_diagnosis_df, "Top 20 Diagnoses for all admissions", save_path + '/diagnosis_distribution_top_20_all.png' if save_path else None)

    #Plot and save the top 20 diagnoses before the first diagnosis
    filtered_admissions_df = add_first_diagnosis_label(admissions_df, filtered_diagnosis_df)
    filtered_admissions_df = filtered_admissions_df[filtered_admissions_df['before_first_diagnosis']]
    filtered_diagnosis_df = diagnosis_df[diagnosis_df['hadm_id'].isin(filtered_admissions_df['hadm_id'])]
    diagnosis_counts = filtered_diagnosis_df['icd_code'].value_counts()
    top_diagnoses_list = diagnosis_counts.head(20).keys().tolist()
    filtered_diagnosis_df = diagnosis_df[diagnosis_df['icd_code'].isin(top_diagnoses_list)]
    filtered_diagnosis_df = filtered_diagnosis_df.sort_values(by='icd_code')
    plot_and_save(filtered_diagnosis_df, "Top 20 Diagnoses before First Diagnosis", save_path + '/diagnosis_distribution_top_20_before.png' if save_path else None)

# Question 8: Is there seasonality in the number of admissions for patients with a certain diagnosis? With a year, month, or day of the week?
def get_admission_seasonality(admissions_df, filtered_diagnosis_df, visualise=False, verbose=True, save_path=None):
    filtered_admissions_df = add_first_diagnosis_label(admissions_df, filtered_diagnosis_df)
    filtered_admissions_df = filtered_admissions_df[filtered_admissions_df['first_diagnosis_label'] == 1]
    filtered_admissions_df['admittime'] = pd.to_datetime(filtered_admissions_df['admittime'])
    filtered_admissions_df['month'] = filtered_admissions_df['admittime'].dt.month
    filtered_admissions_df['day'] = filtered_admissions_df['admittime'].dt.day
    filtered_admissions_df['day_of_week'] = filtered_admissions_df['admittime'].dt.dayofweek
    filtered_admissions_df['year'] = filtered_admissions_df['admittime'].dt.year

    def plot_and_save(column, title, filename):
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_admissions_df[column], kde=True)
        plt.title(title)
        plt.xlabel(column.capitalize())
        plt.ylabel('Count')
        plt.show()
        if filename:
            plt.savefig(filename)

    if visualise:
        plot_and_save('month', 'Admission Month Distribution', save_path + '/admission_month_distribution.png' if save_path else None)
        plot_and_save('day', 'Admission Day Distribution', save_path + '/admission_day_distribution.png' if save_path else None)
        plot_and_save('day_of_week', 'Admission Day of Week Distribution', save_path + '/admission_day_of_week_distribution.png' if save_path else None)
        plot_and_save('year', 'Admission Year Distribution', save_path + '/admission_year_distribution.png' if save_path else None)

    if verbose:
        for column in ['month', 'day', 'day_of_week', 'year']:
            print(f"Admission {column.capitalize()} Distribution:")
            print(filtered_admissions_df[column].value_counts())

# Start Here
# Question 9: What is the length of time between visits/admissions for patients with a certain diagnosis? (top 20)
# Filter this graph for the top-20
# To do: save the values to a log
def get_time_between_visits(admissions_df, filtered_diagnosis_df, visualise=False, verbose=True):
    filtered_admissions_df = add_first_diagnosis_label(admissions_df, filtered_diagnosis_df)
    filtered_admissions_df = filtered_admissions_df[filtered_admissions_df['before_first_diagnosis']]
    all_admissions_before_first_diagnosis = admissions_df[admissions_df['hadm_id'].isin(filtered_diagnosis_df['hadm_id'])]
    filtered_admissions_df['admittime'] = pd.to_datetime(filtered_admissions_df['admittime'])
    filtered_admissions_df['dischtime'] = pd.to_datetime(filtered_admissions_df['dischtime'])
    filtered_admissions_df = filtered_admissions_df.sort_values(by=['subject_id', 'admittime'])
    filtered_admissions_df['next_admittime'] = filtered_admissions_df.groupby('subject_id')['admittime'].shift(-1)
    filtered_admissions_df['time_to_next_admission'] = (filtered_admissions_df['next_admittime'] - filtered_admissions_df['dischtime']).dt.days
    filtered_admissions_df = filtered_admissions_df[filtered_admissions_df['before_first_diagnosis']]

    if visualise:
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_admissions_df['time_to_next_admission'], kde=True, bins=range(0, 100, 1))
        plt.title('Time to Next Admission Distribution of Patients with Diagnosis of Homelessness')
        plt.xlabel('Time to Next Admission')
        plt.ylabel('Count')
        plt.show()
        plt.savefig('/home/rohan/0_homelessness/data/time_to_next_admission.png')
    
    if verbose:
        print(filtered_admissions_df['time_to_next_admission'].describe())
        print(filtered_admissions_df['time_to_next_admission'].value_counts())

# Verified
# What was the type of admission when they were diagnosed with the selected diagnosis?
# To do: save verbosity in log
def get_visit_type_distribution(admissions_df, visualise=False, verbose=True):
    def plot_distribution(df, title, filename):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='admission_type', color='skyblue')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.show()
        plt.savefig(filename)

    subjects_with_more_than_1_visit = admissions_df[admissions_df['subject_id'].duplicated()]['subject_id'].unique()
    
    for label, title, filename in [(1, 'Admission Type Distribution of Patients with Diagnosis of Homelessness', '/home/rohan/0_homelessness/data/admission_type_distribution.png'),
                                   (0, 'Admission Type Distribution of Patients with Diagnosis of Homelessness before First Diagnosis', '/home/rohan/0_homelessness/data/admission_type_distribution_before.png')]:
        filtered_admissions_df = admissions_df[(admissions_df['subject_id'].isin(subjects_with_more_than_1_visit)) & (admissions_df['first_diagnosis_label'] == label)]
        
        if visualise:
            plot_distribution(filtered_admissions_df, title, filename)
        
        if verbose:
            print(filtered_admissions_df['admission_type'].value_counts())

#Verified
# To do: log 
# What was the drug type for all visits before the first diagnosis?
def get_distribution_drug_type(admissions_df, prescriptions_df, visualise=False, verbose=True):
    def process_and_visualize(admissions_df, label, title, filename):
        filtered_admissions_df = admissions_df[(admissions_df['subject_id'].duplicated(keep=False)) & (admissions_df['first_diagnosis_label'] == label)]
        filtered_prescriptions_df = prescriptions_df[prescriptions_df['hadm_id'].isin(filtered_admissions_df['hadm_id'])]
        
        prescription_count = filtered_prescriptions_df['drug'].value_counts()
        top_diagnoses_list = prescription_count.head(20).index.tolist()
        filtered_prescriptions_df = filtered_prescriptions_df[filtered_prescriptions_df['drug'].isin(top_diagnoses_list)]
        
        if visualise:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=filtered_prescriptions_df, x='drug', discrete=True)
            plt.title(title)
            plt.xlabel('Drug')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
            plt.savefig(filename)
        
        if verbose:
            print(prescription_count)
    
    process_and_visualize(admissions_df, 1, 'Number of occurrences of drug types in the dataset at the time of first diagnosis', '/home/rohan/0_homelessness/data/drug_distribution.png')
    process_and_visualize(admissions_df, 0, 'Number of occurrences of drug types in the dataset before first diagnosis', '/home/rohan/0_homelessness/data/drug_distribution_before.png')