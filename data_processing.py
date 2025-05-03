# Complete code with all helper functions to:
# (1) Aggregate diagnoses by time
# (2) Merge with outcomes and demographics
# (3) Build modeling-ready dataset

import pandas as pd

def aggregate_temporal_diagnoses(
    df,
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
    df = df[[patient_id_col, diag_col, date_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    resample_map = {
        'daily': 'D',
        'weekly': 'W-MON',
        'monthly': 'M',
        'quarterly': 'Q'
    }

    if level not in resample_map:
        raise ValueError(f"Unsupported level: {level}")

    df['time_bin'] = df[date_col].dt.to_period(resample_map[level]).dt.start_time

    if agg_type == 'binary':
        df['flag'] = 1
        pivot = df.pivot_table(
            index=[patient_id_col, 'time_bin'],
            columns=diag_col,
            values='flag',
            aggfunc='max',
            fill_value=0
        )
    elif agg_type == 'count':
        pivot = df.pivot_table(
            index=[patient_id_col, 'time_bin'],
            columns=diag_col,
            values=diag_col,
            aggfunc='count',
            fill_value=0
        )
    else:
        raise ValueError("agg_type must be 'binary' or 'count'")

    # Rename columns with consistent prefixes
    pivot.columns = [f"{diag_col}_{level}_{c}" for c in pivot.columns]
    return pivot.reset_index()


def preprocess_outcomes_quarterly(df, patient_id_col='patienticn'):
    """
    Transforms a quarterly outcome file into a long format with [patient_id, time_bin, label].
    
    Assumes columns are like HLDate_2017Q1, HLDate_2017Q2, etc.
    """
    outcome_cols = [col for col in df.columns if col.startswith('HLDate_2017Q')]
    long_df = pd.melt(df, id_vars=[patient_id_col], value_vars=outcome_cols,
                      var_name='quarter', value_name='HL_date')

    long_df['HL_flag'] = long_df['HL_date'].notna().astype(int)
    long_df['time_bin'] = long_df['quarter'].str.extract(r'(2017Q[1-4])')[0]
    long_df['time_bin'] = pd.to_datetime(long_df['time_bin'].str.replace('Q', '-'), errors='coerce').dt.to_period('Q').dt.start_time

    return long_df[[patient_id_col, 'time_bin', 'HL_flag']]


def merge_datasets(
    diag_df,
    outcome_df,
    demo_df=None,
    on_patient_id='patienticn',
    on_time_bin='time_bin',
    join_type='inner'
):
    """
    Merge diagnoses, outcomes, and optionally demographics into a single dataset.

    Parameters:
        diag_df (pd.DataFrame): Aggregated diagnoses (has patient ID + time_bin).
        outcome_df (pd.DataFrame): Outcome labels with same patient ID and time info.
        demo_df (pd.DataFrame): Optional demographics table.
        on_patient_id (str): Column name for patient identifier.
        on_time_bin (str): Column name for time interval (e.g. quarter).
        join_type (str): Join type ('inner', 'left', etc.).

    Returns:
        pd.DataFrame: Final dataset for modeling.
    """
    merged_df = pd.merge(
        diag_df,
        outcome_df,
        on=[on_patient_id, on_time_bin],
        how=join_type
    )

    if demo_df is not None:
        merged_df = pd.merge(
            merged_df,
            demo_df,
            on=on_patient_id,
            how='left'
        )

    return merged_df

