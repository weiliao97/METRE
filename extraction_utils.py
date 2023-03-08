import pandas as pd
import numpy as np
'''
Some Util funcs adapted from MIMIC-Extract based on the new features of MIMIC-IV 
https://github.com/MLforHealth/MIMIC_Extract
Wang et al.. MIMIC-Extract: A Data Extraction, Preprocessing, and Representation 
Pipeline for MIMIC-III. arXiv:1907.08322. 
'''
ID_COLS = ['subject_id', 'hadm_id', 'stay_id']
ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']


def combine_cols(col_1, col_2):
    """
    Combine columns from different itemids but same measurement
    :param col_1: columns to be removed after merging
    :param col_2: columns to be kept after merging
    :return: col_2:
    """
    col_2.columns.names = ['LEVEL2', 'Aggregation Function']
    col_1 = col_1.droplevel(level=0, axis=1)
    col_2 = col_2.droplevel(level=0, axis=1)

    # or_filled = len(b) - b.loc[:, 'mean'].isnull().sum()
    # or_mean = b.loc[:, 'mean'].dropna().mean()

    row_mask = (col_1.loc[:, 'count'] > 0).values
    mask = (col_2.loc[:, 'count'] > 0).values

    c = col_2.loc[row_mask * mask, 'count'].mul(col_2.loc[row_mask * mask, 'mean'].values) + \
        col_1.loc[row_mask * mask, 'count'].mul(col_1.loc[row_mask * mask, 'mean'].values)
    d = col_2.loc[row_mask * mask, 'count'] + col_1.loc[row_mask * mask, 'count']

    col_2.loc[row_mask * mask, 'mean'] = c / d
    col_2.loc[:, 'count'] = col_2.loc[:, 'count'] + col_1.loc[:, 'count']

    col_2.loc[~mask, 'mean'] = col_1.loc[~mask, 'mean']

    # c_filled = len(b) - b.loc[:, 'mean'].isnull().sum()
    # c_mean = b.loc[:, 'mean'].dropna().mean()

    # print('Original mean is %.3f, original filled is %d\nCombined mean is %.3f, combined filled is %d\n' % (
    # or_mean, or_filled, c_mean, c_filled))
    return col_2

def range_unnest(df, col, out_col_name=None, reset_index=False):
    """
    Create multiple rows for a stay based on max stay hours
    :param df: pd.DataFrame, a dataframe contains basic patient demographics, shape e.g. (38766, 20)
    :param col: str, column name to be unfolded, e.g. 'max_hours'
    :param out_col_name: str, e.g. 'hours_in'
    :param reset_index: bool, whether set index or not
    :return: col_flat, pd.DataFrame, create rows for each stay_id based on 'max_hours', shape e.g.  (2697900, 2)
    """
    assert len(df.index.names) == 1, "Does not support multi-index."
    if out_col_name is None:
        out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(y + 1)],
        columns=[df.index.names[0], out_col_name]
    )

    if not reset_index:
        col_flat = col_flat.set_index(df.index.names[0])
    return col_flat

def process_query_results(df, fill_df):
    """
    :param df: pd.DataFrame, results after query, shape e.g. (237081, 27)
    :param fill_df: pd.DataFrame, a multiindex template, indices: subject_id hadm_id stay_id hours_in,
                    shape e.g. (2697900, 1)
    :return: df: pd.DataFrame, dataframe after filling query results into the template and
                apply aggregation function, shape e.g. (2697900, 44), with row index same as fill_df
                but column becomes a multiindex, e.g. level 0: so2, level 1: mean, count
    """
    df = df.groupby(ID_COLS + ['hours_in']).agg(['mean', 'count'])
    df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)
    df = df.reindex(fill_df.index)
    return df

def compile_intervention(inv_query, c, time_window=1):
    """
    Organize queried intervention table
    :param inv_query: pd.DataFrame, Queried intervention results, shape e.g.  (54257, 8)
                    columns, e.g. subject_id, hadm_id, stay_id, starttime, endtime, icu_intime, icu_outtime, max_hours
    :param c: str, column name of the intervention procedure, e.g. 'vent'
    :return: inv_query: pd.DataFrame, after organizing, the last column will indicate a state at that hour (0 or 1)
                    columns, e.g. stay_id, subject_id, hadm_id, hours_in, vent, shape e.g. (2290028, 5)
    """
    # df_copy = df.copy(deep=True)
    to_hours = lambda x: max(0, x.days * 24//time_window + x.seconds // (3600 * time_window))
    inv_query['max_hours'] = (inv_query['icu_outtime'] - inv_query['icu_intime']).apply(to_hours)
    inv_query.loc[:, 'starttime'] = inv_query.loc[:, ['starttime', 'icu_intime']].max(axis=1)
    inv_query.loc[:, 'endtime'] = inv_query.loc[:, ['endtime', 'icu_outtime']].min(axis=1)
    inv_query['starttime'] = inv_query['starttime'] - inv_query['icu_intime']
    inv_query['starttime'] = inv_query.starttime.apply(to_hours) #lambda x: x.days * 24 + x.seconds // 3600)
    inv_query['endtime'] = inv_query['endtime'] - inv_query['icu_intime']
    inv_query['endtime'] = inv_query.endtime.apply(to_hours) #lambda x: x.days * 24 + x.seconds // 3600)
    if c == 'antibiotics':
        inv_query = inv_query.groupby('stay_id').apply(add_antibitics_indicators)
    else:
        inv_query = inv_query.groupby('stay_id').apply(add_outcome_indicators)

    inv_query.rename(columns={'on': c}, inplace=True)
    # heparin_2.rename(columns={'values': c + ' conc'}, inplace=True)
    inv_query = inv_query.reset_index(level='stay_id')
    return inv_query

def add_outcome_indicators(out_gb):
    """
    Iterate a groupby object and add intervention procedure indicator
    :param out_gb: Pandas groupby object, specific intervention variable grouped by e.g. 'stay_id'
    :return: pd.DataFrame, for each stay_id, iterate through the hours with the procedure, represent it by 1
                for any other hours, fill 0, shape e.g. (2290028, 5)
    """
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    # icustay_id = out_gb['stay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()
    # on_values = []

    # p_set = on_hrs.copy()
    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))
        # if on_hrs - p_set:
        # only when sets updates, append a value
        # on_values.append([row['values']]*len(on_hrs - p_set))
        # p_set = on_hrs.copy()

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    # values flatten a nested list
    # values = [0]*len(off_hrs) + [item for sublist in on_values for item in sublist]
    on_vals = [0] * len(off_hrs) + [1] * len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id': hadm_id,
                         'hours_in': hours, 'on': on_vals})  # icustay_id': icustay_id})

def add_antibitics_indicators(out_gb):
    """
    Iterate a groupby object and add intervention procedure indicator --- antibiotics version
    :param out_gb: Pandas groupby object, specific intervention variable grouped by e.g. 'stay_id'
    :return: pd.DataFrame, for each stay_id, iterate through the hours with the antibitics name and route
            shape e.g. (1964815, 6), column names: stay_id, subject_id, hadm_id, hours_in, antibiotic, route
    """
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    # icustay_id = out_gb['stay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()
    on_values = []
    on_route = []

    p_set = on_hrs.copy()
    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))
        if on_hrs - p_set:
            # only when sets updates, append a value
            on_values.append([row['antibiotic']] * len(on_hrs - p_set))
            on_route.append([row['route']] * len(on_hrs - p_set))

        p_set = on_hrs.copy()

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    # values flatten a nested list
    values = [np.nan] * len(off_hrs) + [item for sublist in on_values for item in sublist]
    route = [np.nan] * len(off_hrs) + [item for sublist in on_route for item in sublist]
    # on_vals = [0]*len(off_hrs) + [1]*len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id': hadm_id,
                         'hours_in': hours, 'antibiotic': values, 'route': route})  # icustay_id': icustay_id})

def add_blank_indicators(out_gb):
    """
    Function to add blank indicator for stays with no ventilation records
    :param out_gb: Pandas groupby object, grouped from a dataframe
    :return: pd.DataFrame, shape e.g. (407872, 4), index: stay_id, column names: subject_id, hadm_id, hours_in, on
            with on being all 0s
    """
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    # icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]

    hrs = range(max_hrs + 1)
    vals = list([0] * len(hrs))
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id': hadm_id,
                         'hours_in': hrs, 'on': vals})  # 'icustay_id': icustay_id,


def continuous_outcome_processing(out_data, data, icustay_timediff):
    '''

    :param out_data:
    :param data:
    :param icustay_timediff:
    :return:
    '''
    to_hours = lambda x: max(0, x.days * 24 // time_window + x.seconds // (3600 * time_window))
    out_data['icu_intime'] = out_data['stay_id'].map(data['icu_intime'].to_dict())
    out_data['icu_outtime'] = out_data['stay_id'].map(data['icu_outtime'].to_dict())
    out_data['max_hours'] = out_data['stay_id'].map(icustay_timediff)
    out_data['starttime'] = out_data['starttime'] - out_data['icu_intime']
    out_data['starttime'] = out_data.starttime.apply(to_hours)  #lambda x: x.days * 24 + x.seconds // 3600)
    out_data['endtime'] = out_data['endtime'] - out_data['icu_intime']
    out_data['endtime'] = out_data.endtime.apply(to_hours) #lambda x: x.days * 24 + x.seconds // 3600)
    out_data = out_data.groupby(['stay_id'])
    return out_data

def remove_outliers_h(X, X_or, col, range):
    """
    Remove entries higher than a threshold and set count column to 0
    :param X: pd.DataFrame, all the columns with mean columns to be changed
    :param X_or: pd.DataFrame, all the mean columns with original values
    :param col: str, name of the column to be cleaned
    :param range: int, the threshold value
    :return: None,
    """
    X.loc[:, [(col, 'mean')]] = X.loc[:, [(col, 'mean')]].mask((X_or.loc[:, [(col, 'mean')]] > range).values)
    X.loc[:, [(col, 'count')]] = X.loc[:, [(col, 'count')]].mask((X_or.loc[:, [(col, 'mean')]] > range).values, other=0.0)
    return

def remove_outliers_l(X, X_or, col, range):
    """
    Remove entries lower than a threshold and set count column to 0
    :param X: pd.DataFrame, all the columns with mean columns to be changed
    :param X_or: pd.DataFrame, all the mean columns with original values
    :param col: str, name of the column to be cleaned
    :param range: int, the threshold value
    :return: None,
    """
    X.loc[:, [(col, 'mean')]] = X.loc[:, [(col, 'mean')]].mask((X_or.loc[:, [(col, 'mean')]] < range).values)
    X.loc[:, [(col, 'count')]] = X.loc[:, [(col, 'count')]].mask((X_or.loc[:, [(col, 'mean')]] < range).values, other=0.0)
    return

def fill_query(df, fill_df, tw_in_min, time='chartoffset'):
    """
    Organize queried results into the template fill_df format
    :param df: pd.DataFrame, queried results
    :param fill_df: pd.DataFrame, a template with all stay ids and each stay id has hours_in level 1 index
    :param time: str, how the time column was represented e.g. 'chartoffset', 'observationoffset'
    :return: df, pd.DataFrame, the same structure as the template
    """
    df['hours_in'] = df[time].floordiv(tw_in_min)
    df.drop(columns=[time], inplace=True)
    df.set_index(['patientunitstayid']+ ['hours_in'], inplace=True)
    df.reset_index(inplace=True)
    # level_to_change = 1
    df = df.groupby(['patientunitstayid'] + ['hours_in']).agg(['mean', 'count'])
    # df.index = df.index.set_levels(df.index.levels[level_to_change].astype(int), level=level_to_change)
    df = df.reindex(fill_df.index)
    return df

def add_outcome_indicators_e(out_gb):
    """
    For eICU data, iterate a groupby object and add intervention procedure indicator
    :param out_gb: Pandas groupby object, specific intervention variable grouped by e.g. 'patientunitstayid'
    :return: pd.DataFrame, for each stay_id, iterate through the hours with the procedure, represent it by 1
                for any other hours, fill 0,
    """
    # patientunitstayid = out_gb['patientunitstayid'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()
    # on_values = []

    p_set = on_hrs.copy()
    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))
        # if on_hrs - p_set:
        # only when sets updates, append a value
            # on_values.append([row['values']]*len(on_hrs - p_set))
        # p_set = on_hrs.copy()

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    ##values flatten a nested list
    # values = [0]*len(off_hrs) + [item for sublist in on_values for item in sublist]
    on_vals = [0]*len(off_hrs) + [1]*len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({
                        'hours_in':hours, 'on':on_vals}) #icustay_id': icustay_id})

def add_blank_indicators_e(out_gb):
    """
    For eICU data, add blank indicator for stays with no ventilation records
    :param out_gb: Pandas groupby object, grouped from a dataframe
    :return: pd.DataFrame, index: patientunitstayid, column names: hours_in, on
            with on being all 0s
    """
    # subject_id = out_gb['subject_id'].unique()[0]
    # hadm_id = out_gb['hadm_id'].unique()[0]
    #icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]

    hrs = range(max_hrs + 1)
    vals = list([0]*len(hrs))
    return pd.DataFrame({
                        'hours_in':hrs, 'on':vals})#'icustay_id': icustay_id,

def process_inv(df, name):
    """
    Organize queried intervention table
    :param df:  pd.DataFrame, Queried intervention results,
    :param name: str, column name of the intervention procedure, e.g. 'vent'
    :return: df, pd.DataFrame, after organizing, the last column will indicate a state at that hour (0 or 1)
                    columns, e.g. patientunitstayid, hours_in, vent
    """
    df.starttime = df.starttime.astype(int)
    df.endtime = df.endtime.astype(int)
    df.max_hours = df.max_hours.astype(int)
    df = df.groupby(['patientunitstayid']).apply(add_outcome_indicators_e)
    df.rename(columns={'on': name}, inplace=True)
    df = df.reset_index()
    return df
