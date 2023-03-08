# Set up Google big query
from google.colab import auth
from google.cloud import bigquery
import os
import json
import pickle
import numpy as np
import pandas as pd
from extraction_utils import *
from extract_sql import *

auth.authenticate_user()


def extract_mimic(args):
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    client = bigquery.Client(project=args.project_id)
    # MIMIC-IV id
    ID_COLS = ['subject_id', 'hadm_id', 'stay_id']
    # datatime format to hour
    to_hours = lambda x: max(0, x.days * 24//args.time_window + x.seconds // (3600 * args.time_window))

    # get group id, could be sepsis3, ARF, shock, COPD, CHF
    patient = get_patient_group(args, client)
    print("Patient icu info query done, start querying variables in Dynamic table")
    # get icu stay id and subject id
    icuids_to_keep = patient['stay_id']
    icuids_to_keep = set([str(s) for s in icuids_to_keep])
    subject_to_keep = patient['subject_id']
    subject_to_keep = set([str(s) for s in subject_to_keep])
    # create template fill_df with time window for each stay based on icu in/out time
    patient.set_index('stay_id', inplace=True)
    patient['max_hours'] = (patient['icu_outtime'] - patient['icu_intime']).apply(to_hours)
    missing_hours_fill = range_unnest(patient, 'max_hours', out_col_name='hours_in', reset_index=True)
    missing_hours_fill['tmp'] = np.NaN
    fill_df = patient.reset_index()[ID_COLS].join(missing_hours_fill.set_index('stay_id'), on='stay_id')
    fill_df.set_index(ID_COLS + ['hours_in'], inplace=True)

    # start with mimic_derived_data
    # query bg table
    bg = query_bg_mimic(client, subject_to_keep)
    # initial process bg table
    bg['hours_in'] = (bg['charttime'] - bg['icu_intime']).apply(to_hours)
    bg.drop(columns=['charttime', 'icu_intime', 'aado2_calc'], inplace=True) # aado2_calc is not used
    bg = process_query_results(bg, fill_df)

    # query vital sign
    vitalsign = query_vitals_mimic(client, icuids_to_keep)
    # temperature/glucose is a repeat name but different itemid, rename for now and combine later
    vitalsign.rename(columns={'temperature': 'temp_vital'}, inplace=True)
    vitalsign.rename(columns={'glucose': 'glucose_vital'}, inplace=True)
    vitalsign['hours_in'] = (vitalsign['charttime'] - vitalsign['icu_intime']).apply(to_hours)
    vitalsign.drop(columns=['charttime', 'icu_intime', 'temperature_site'], inplace=True) # temperature_site is not used
    vitalsign = process_query_results(vitalsign, fill_df)

    # query blood differential
    blood_diff = query_blood_diff_mimic(client, subject_to_keep)
    blood_diff['hours_in'] = (blood_diff['charttime'] - blood_diff['icu_intime']).apply(to_hours)
    blood_diff.drop(columns=['charttime', 'icu_intime', 'specimen_id'], inplace=True)
    blood_diff = process_query_results(blood_diff, fill_df)

    # query cardiac marker
    cardiac_marker = query_cardiac_marker_mimic(client, subject_to_keep)
    cardiac_marker['troponin_t'].replace(to_replace=[None], value=np.nan, inplace=True)
    cardiac_marker['troponin_t'] = pd.to_numeric(cardiac_marker['troponin_t'])
    cardiac_marker['hours_in'] = (cardiac_marker['charttime'] - cardiac_marker['icu_intime']).apply(to_hours)
    cardiac_marker.drop(columns=['charttime', 'icu_intime', 'specimen_id'], inplace=True)
    cardiac_marker = process_query_results(cardiac_marker, fill_df)

    # query chemistry
    chemistry = query_chemistry_mimic(client, subject_to_keep)
    # rename glucose into glucose_chem and others
    chemistry.rename(columns={'glucose': 'glucose_chem'}, inplace=True)
    chemistry.rename(columns={'bicarbonate': 'bicarbonate_chem'}, inplace=True)
    chemistry.rename(columns={'chloride': 'chloride_chem'}, inplace=True)
    chemistry.rename(columns={'calcium': 'calcium_chem'}, inplace=True)
    chemistry.rename(columns={'potassium': 'potassium_chem'}, inplace=True)
    chemistry.rename(columns={'sodium': 'sodium_chem'}, inplace=True)
    chemistry['hours_in'] = (chemistry['charttime'] - chemistry['icu_intime']).apply(to_hours)
    chemistry.drop(columns=['charttime', 'icu_intime', 'specimen_id'], inplace=True)
    chemistry = process_query_results(chemistry, fill_df)

    # query coagulation
    coagulation = query_coagulation_mimic(client, subject_to_keep)
    coagulation['hours_in'] = (coagulation['charttime'] - coagulation['icu_intime']).apply(to_hours)
    coagulation.drop(columns=['charttime', 'icu_intime', 'specimen_id'], inplace=True)
    coagulation = process_query_results(coagulation, fill_df)

    # query cbc
    cbc = query_cbc_mimic(client, subject_to_keep)
    cbc.rename(columns={'hematocrit': 'hematocrit_cbc'}, inplace=True)
    cbc.rename(columns={'hemoglobin': 'hemoglobin_cbc'}, inplace=True)
    # also drop wbc since it's a repeat 51301
    cbc['hours_in'] = (cbc['charttime'] - cbc['icu_intime']).apply(to_hours)
    cbc.drop(columns=['charttime', 'icu_intime', 'specimen_id', 'wbc'], inplace=True)
    cbc = process_query_results(cbc, fill_df)

    # query culture
    culture = query_culture_mimic(client, subject_to_keep)
    culture.rename(columns={'specimen': 'specimen_culture'}, inplace=True)
    culture['hours_in'] = (culture['charttime'] - culture['icu_intime']).apply(to_hours)
    culture.drop(columns=['charttime', 'icu_intime'], inplace=True)
    culture = culture.groupby(ID_COLS + ['hours_in']).agg(['last'])
    culture = culture.reindex(fill_df.index)

    # query enzyme
    enzyme = query_enzyme_mimic(client, subject_to_keep)
    # also drop ck_mb since it's a repeat 50911
    enzyme['hours_in'] = (enzyme['charttime'] - enzyme['icu_intime']).apply(to_hours)
    enzyme.drop(columns=['charttime', 'icu_intime', 'specimen_id', 'ck_mb'], inplace=True)
    enzyme = process_query_results(enzyme, fill_df)

    # query gcs
    gcs = query_gcs_mimic(client, icuids_to_keep)
    gcs['hours_in'] = (gcs['charttime'] - gcs['icu_intime']).apply(to_hours)
    gcs.drop(columns=['charttime', 'icu_intime'], inplace=True)
    gcs = process_query_results(gcs, fill_df)

    # query inflammation
    inflammation = query_inflammation_mimic(client, subject_to_keep)
    inflammation['hours_in'] = (inflammation['charttime'] - inflammation['icu_intime']).apply(to_hours)
    inflammation.drop(columns=['charttime', 'icu_intime'], inplace=True)
    inflammation = process_query_results(inflammation, fill_df)

    # query uo
    uo = query_uo_mimic(client, icuids_to_keep)
    uo['hours_in'] = (uo['charttime'] - uo['icu_intime']).apply(to_hours)
    uo.drop(columns=['charttime', 'icu_intime'], inplace=True)
    uo = process_query_results(uo, fill_df)

    # join and save
    # use MIMIC-Extract way to query other itemids that was present in MIMIC-Extract
    # load resources
    chartitems_to_keep = pd.read_excel('./resources/chartitems_to_keep_0505.xlsx')
    lab_to_keep = pd.read_excel('./resources/labitems_to_keep_0505.xlsx')
    var_map = pd.read_csv('./resources/Chart_makeup_0505 - var_map0505.csv')
    chart_items = chartitems_to_keep['chartitems_to_keep'].tolist()
    lab_items = lab_to_keep['labitems_to_keep'].tolist()
    chart_items = set([str(i) for i in chart_items])
    lab_items = set([str(i) for i in lab_items])

    # additional chart and lab
    chart_lab = query_chart_lab_mimic(client, icuids_to_keep, chart_items, lab_items)
    chart_lab['value'] = pd.to_numeric(chart_lab['value'], 'coerce')
    chart_lab = chart_lab.set_index('stay_id').join(patient[['icu_intime']])
    chart_lab['hours_in'] = (chart_lab['charttime'] - chart_lab['icu_intime']).apply(to_hours)
    chart_lab.drop(columns=['charttime', 'icu_intime'], inplace=True)
    chart_lab.set_index('itemid', append=True, inplace=True)
    var_map.set_index('itemid', inplace=True)
    chart_lab = chart_lab.join(var_map, on='itemid').set_index(['LEVEL1', 'LEVEL2'], append=True)
    chart_lab.index.names = ['stay_id', chart_lab.index.names[1], chart_lab.index.names[2], chart_lab.index.names[3]]
    group_item_cols = ['LEVEL2']
    chart_lab = chart_lab.groupby(ID_COLS + group_item_cols + ['hours_in']).agg(['mean', 'count'])

    chart_lab.columns = chart_lab.columns.droplevel(0)
    chart_lab.columns.names = ['Aggregation Function']
    chart_lab = chart_lab.unstack(level=group_item_cols)
    chart_lab.columns = chart_lab.columns.reorder_levels(order=group_item_cols + ['Aggregation Function'])

    chart_lab = chart_lab.reindex(fill_df.index)
    chart_lab = chart_lab.sort_index(axis=1, level=0)
    new_cols = chart_lab.columns.reindex(['mean', 'count'], level=1)
    chart_lab = chart_lab.reindex(columns=new_cols[0])

    # join all dataframes
    total = bg.join(
        [vitalsign, blood_diff, cardiac_marker, chemistry, coagulation, cbc, culture, enzyme, gcs, inflammation, uo])

    # start combining and drop some columns either due to redundancy or not well-populated
    # drop some columns (not well-populated or already dependent on existing columns )
    columns_to_drop = ['rdwsd', 'aado2', 'pao2fio2ratio', 'carboxyhemoglobin',
                       'methemoglobin', 'globulin', 'd_dimer', 'thrombin', 'basophils_abs', 'eosinophils_abs',
                       'lymphocytes_abs', 'monocytes_abs', 'neutrophils_abs']
    for c in columns_to_drop:
        total.drop(c, axis=1, level=0, inplace=True)

    idx = pd.IndexSlice
    chart_lab.loc[:, idx[:, ['count']]] = chart_lab.loc[:, idx[:, ['count']]].fillna(0)
    total.loc[:, idx[:, ['count']]] = total.loc[:, idx[:, ['count']]].fillna(0)

    # combine columns since they were from different itemids but have the same semantics
    names_to_combine = [
        ['so2', 'spo2'], ['fio2', 'fio2_chartevents'], ['bicarbonate', 'bicarbonate_chem'],
        ['hematocrit', 'hematocrit_cbc'], ['hemoglobin', 'hemoglobin_cbc'], ['chloride', 'chloride_chem'],
        ['glucose', 'glucose_chem'], ['glucose', 'glucose_vital'],
        ['temperature', 'temp_vital'], ['sodium', 'sodium_chem'], ['potassium', 'potassium_chem']
    ]
    for names in names_to_combine:
        original = total.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
        makeups = total.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
        filled = combine_cols(makeups, original)
        total.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
        total.drop(names[1], axis=1, level=0, inplace=True)

    with open('./json_files/mimic_culturesite_map.json') as f:
        csite_map = json.load(f)
    total.loc[:, idx['specimen_culture', ['last']]] = pd.Series(
        np.squeeze(total.loc[:, idx['specimen_culture', ['last']]].values)).map(csite_map).values

    # drop Eosinophils
    chart_lab.drop('Eosinophils', axis=1, level=0, inplace=True)
    # combine in chart_lab table
    names = ['Phosphate', 'Phosphorous']
    original = chart_lab.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
    makeups = chart_lab.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    chart_lab.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
    chart_lab.drop(names[1], axis=1, level=0, inplace=True)

    names = ['Potassium', 'Potassium serum']
    original = chart_lab.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
    makeups = chart_lab.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    chart_lab.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
    chart_lab.drop(names[1], axis=1, level=0, inplace=True)

    # Combine between chartlab and total table
    with open('./json_files/mimic_to_combine_1.json') as f:
        names_list = json.load(f)
    for names in names_list:
        original = total.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
        makeups = chart_lab.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
        filled = combine_cols(makeups, original)
        total.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
        chart_lab.drop(names[1], axis=1, level=0, inplace=True)

    # In eicu mbp contains both invasive and non-invasive, so combine them for mimic_iv
    names_list = [['dbp', 'Diastolic blood pressure'], ['dbp_ni', 'Diastolic blood pressure'],
                  ['mbp', 'Mean blood pressure'], ['mbp_ni', 'Mean blood pressure'],
                  ['sbp', 'Systolic blood pressure'], ['sbp_ni', 'Systolic blood pressure']]
    for names in names_list:
        original = total.loc[:, idx[names[0], ['count', 'mean']]].copy(deep=True)
        makeups = chart_lab.loc[:, idx[names[1], ['count', 'mean']]].copy(deep=True)
        filled = combine_cols(makeups, original)
        total.loc[:, idx[names[0], ['count', 'mean']]] = filled.loc[:, ['count', 'mean']].values
        # Xm.drop(names[1], axis=1, level=0, inplace=True)
    chart_lab.drop('Mean blood pressure', axis=1, level=0, inplace=True)
    chart_lab.drop('Diastolic blood pressure', axis=1, level=0, inplace=True)
    chart_lab.drop('Systolic blood pressure', axis=1, level=0, inplace=True)

    with open('./json_files/mimic_to_drop_1.json') as f:
        columns_to_drop = json.load(f)
    for c in columns_to_drop:
        chart_lab.drop(c, axis=1, level=0, inplace=True)
    vital = total.join(chart_lab)
    # Done dropping and combining

    # screen and positive culture needs impute, they are last columns but with float data type
    vital_encode = pd.get_dummies(vital)

    vital_encode[('positive_culture', 'mask')] = (~vital_encode[('positive_culture', 'last')].isnull()).astype(float)
    vital_encode[('screen', 'mask')] = (~vital_encode[('screen', 'last')].isnull()).astype(float)
    vital_encode[('has_sensitivity', 'mask')] = (~vital_encode[('has_sensitivity', 'last')].isnull()).astype(float)
    # X_encode.fillna(value=0, inplace=True)
    # vital_encode.fillna(value=0, inplace=True)

    col = vital_encode.columns.to_list()
    col.insert(col.index(('screen', 'last')) + 1, ('screen', 'mask'))
    col.insert(col.index(('positive_culture', 'last')) + 1, ('positive_culture', 'mask'))
    col.insert(col.index(('has_sensitivity', 'last')) + 1, ('has_sensitivity', 'mask'))

    vital_final = vital_encode[col[:-3]]

    # check if any culture site is missing and fill in empty
    col_encode = vital_final.columns.to_list()
    csite_col = [int(i.split('cul_site')[-1]) for i in col_encode if "cul_site" in i]
    if len(csite_col) < 14:
        # find out which is missing
        missing_site = [i for i in range(14) if i not in csite_col]
        missing_col_name = ["('specimen_culture', 'last')_cul_site" + str(i) for i in missing_site]
        for col in missing_col_name:
            vital_final[col] = 0
    with open('./json_files/mimic_col_order.pickle', 'rb') as f:
        mimic_col_order = pickle.load(f)
    vital_final = vital_final[mimic_col_order]
    print('Start querying variables in the Intervention table')
    ####### Done vital table #######

    # start query intervention
    vent_data = query_vent_mimic(client, icuids_to_keep)
    vent_data = compile_intervention(vent_data, 'vent', args.time_window)

    ids_with = vent_data['stay_id']
    ids_with = set(map(int, ids_with))
    ids_all = set(map(int, icuids_to_keep))
    ids_without = (ids_all - ids_with)
    novent_data = patient.copy(deep=True)
    novent_data = novent_data.reset_index()
    novent_data = novent_data.set_index('stay_id')
    novent_data = novent_data.iloc[novent_data.index.isin(ids_without)]
    novent_data = novent_data.reset_index()
    novent_data = novent_data[['subject_id', 'hadm_id', 'stay_id', 'max_hours']]
    # novent_data['max_hours'] = novent_data['stay_id'].map(icustay_timediff)
    novent_data = novent_data.groupby('stay_id')
    novent_data = novent_data.apply(add_blank_indicators)
    novent_data.rename(columns={'on': 'vent'}, inplace=True)
    novent_data = novent_data.reset_index()

    # Concatenate all the data vertically
    intervention = pd.concat([vent_data[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'vent']],
                              novent_data[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'vent']]],
                             axis=0)

    # query antibiotics
    antibiotics = query_antibiotics_mimic(client, icuids_to_keep)
    antibiotics = compile_intervention(antibiotics, 'antibiotics', args.time_window)
    intervention = intervention.merge(
        antibiotics[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'antibiotic', 'route']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # vaso agents
    column_names = ['dopamine', 'epinephrine', 'norepinephrine', 'phenylephrine', 'vasopressin', 'dobutamine',
                    'milrinone']
    for c in column_names:
        # TOTAL VASOPRESSOR DATA
        new_data = query_vasoactive_mimic(client, icuids_to_keep, c)
        new_data = compile_intervention(new_data, c, args.time_window)
        intervention = intervention.merge(
            new_data[['subject_id', 'hadm_id', 'stay_id', 'hours_in', c]],
            on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
            how='left'
        )

    # heparin
    heparin = query_heparin_mimic(client, subject_to_keep)
    heparin = compile_intervention(heparin, 'heparin', args.time_window)
    intervention = intervention.merge(
        heparin[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'heparin']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # crrt
    crrt = query_crrt_mimic(client, icuids_to_keep)
    crrt = compile_intervention(crrt, 'crrt', args.time_window)
    intervention = intervention.merge(
        crrt[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'crrt']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # rbc transfusion
    rbc_trans = query_rbc_trans_mimic(client, icuids_to_keep)
    rbc_trans = compile_intervention(rbc_trans, 'rbc_trans', args.time_window)
    intervention = intervention.merge(
        rbc_trans[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'rbc_trans']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # platelets transfusion
    platelets_trans = query_pll_trans_mimic(client, icuids_to_keep)
    platelets_trans = compile_intervention(platelets_trans, 'platelets_trans', args.time_window)
    intervention = intervention.merge(
        platelets_trans[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'platelets_trans']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # ffp transfusion
    ffp_trans = query_ffp_trans_mimic(client, icuids_to_keep)
    ffp_trans = compile_intervention(ffp_trans, 'ffp_trans', args.time_window)
    intervention = intervention.merge(
        ffp_trans[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'ffp_trans']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # other infusion
    colloid_bolus = query_colloid_mimic(client, icuids_to_keep)
    colloid_bolus = compile_intervention(colloid_bolus, 'colloid_bolus', args.time_window)
    intervention = intervention.merge(
        colloid_bolus[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'colloid_bolus']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # other infusion
    crystalloid_bolus = query_crystalloid_mimic(client, icuids_to_keep)
    crystalloid_bolus = compile_intervention(crystalloid_bolus, 'crystalloid_bolus', args.time_window)
    intervention = intervention.merge(
        crystalloid_bolus[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'crystalloid_bolus']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left')

    # Process the Intervention table
    intervention.drop('route', axis=1, inplace=True) # drop route column
    # for each column, astype to int and fill na with 0
    intervention = intervention.fillna(0)
    intervention.loc[:, 'antibiotic'] = intervention.loc[:, 'antibiotic'].mask(intervention.loc[:, 'antibiotic'] != 0,
                                                                               1).values
    for i in range(5, 20):
        intervention.iloc[:, i] = intervention.iloc[:, i].astype(int)
    intervention.set_index(ID_COLS + ['hours_in'], inplace=True)
    intervention.sort_index(level=['stay_id', 'hours_in'], inplace=True)
    # Finish processing the Intervention table
    print('Start querying variables in the Static table')

    # static info
    #  query patients anchor year and comorbidity
    anchor_year = query_anchor_year_mimic(client, icuids_to_keep)
    comorbidity = query_comorbidity_mimic(client, icuids_to_keep)
    patient.reset_index(inplace=True)
    patient.set_index(ID_COLS, inplace=True)
    comorbidity.set_index(ID_COLS, inplace=True)
    anchor_year.set_index(ID_COLS, inplace=True)
    static = patient.join([comorbidity, anchor_year['anchor_year_group']])

    if args.exit_point == 'Raw':
        print('Exit point is after querying raw records, saving results...')
        vital_final.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_vital.h5'), key='mimic_vital')
        static.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_static.h5'), key='mimic_static')
        intervention.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_inv.h5'), key='mimic_inv')
        return

    # remove outliers
    total_cols = vital_final.columns.tolist()
    mean_col = [i for i in total_cols if 'mean' in i]
    X_mean = vital_final.loc[:, mean_col]

    if not args.no_removal:
        print('Performing outlier removal')
        with open("./json_files/mimic_outlier_high.json") as f:
            range_dict_high  = json.load(f)
        with open("./json_files/mimic_outlier_low.json") as f:
            range_dict_low = json.load(f)
        for var_to_remove in range_dict_high:
            remove_outliers_h(vital_final, X_mean, var_to_remove, range_dict_high[var_to_remove])
        for var_to_remove in range_dict_low:
            remove_outliers_l(vital_final, X_mean, var_to_remove, range_dict_low[var_to_remove])
    else:
        print('Skipped outlier removal')

    if args.exit_point == 'Outlier_removal':
        print('Exit point is after removing outliers, saving results...')
        vital_final.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_vital.h5'), key='mimic_vital')
        static.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_static.h5'), key='mimic_static')
        intervention.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_inv.h5'), key='mimic_inv')
        return

    # normalize
    print('Start normalization and data imputation ')
    count_col = [i for i in total_cols if 'count' in i]
    col_means, col_stds = vital_final.loc[:, mean_col].mean(axis=0), vital_final.loc[:, mean_col].std(axis=0)
    # saving col_means and col_stds for eicu normalization
    df_mean_std = col_means.to_frame('mean').join(col_stds.to_frame('std'))
    df_mean_std.to_hdf(os.path.join(args.output_dir, 'MIMIC_mean_std_stats.h5'), 'MIMIC_mean_std')
    vital_final.loc[:, mean_col] = (vital_final.loc[:, mean_col] - col_means) / col_stds
    icustay_means = vital_final.loc[:, mean_col].groupby(ID_COLS).mean()
    # impute
    vital_final.loc[:, mean_col] = vital_final.loc[:, mean_col].groupby(ID_COLS).fillna(method='ffill').groupby(
        ID_COLS).fillna(
        icustay_means).fillna(0)
    # 0 or 1
    vital_final.loc[:, count_col] = (vital_final.loc[:, count_col] > 0).astype(float)
    # at this satge only 3 last columns has nan values
    vital_final = vital_final.fillna(0)
    # convert to int in stead of int64 which will be problematic for hdf saving
    vital_final[('screen', 'last')] = vital_final[('screen', 'last')].astype('uint8')
    vital_final[('positive_culture', 'last')] = vital_final[('positive_culture', 'last')].astype('uint8')
    vital_final[('has_sensitivity', 'last')] = vital_final[('has_sensitivity', 'last')].astype('uint8')

    if args.exit_point == 'Impute':
        print('Exit point is after data imputation, saving results...')
        vital_final.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_vital.h5'), key='mimic_vital')
        static.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_static.h5'), key='mimic_static')
        intervention.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_inv.h5'), key='mimic_inv')
        return

    # split data
    stays_v = set(vital_final.index.get_level_values(2).values)
    stays_static = set(static.index.get_level_values(2).values)
    stays_int = set(intervention.index.get_level_values(2).values)
    assert stays_v == stays_static, "Subject ID pools differ!"
    assert stays_v == stays_int, "Subject ID pools differ!"
    train_frac, dev_frac, test_frac = 0.7, 0.1, 0.2
    SEED = 41
    np.random.seed(SEED)
    subjects, N = np.random.permutation(list(stays_v)), len(stays_v)
    N_train, N_dev, N_test = int(train_frac * N), int(dev_frac * N), int(test_frac * N)
    train_stay = list(stays_v)[:N_train]
    dev_stay = list(stays_v)[N_train:N_train + N_dev]
    test_stay = list(stays_v)[N_train + N_dev:]
    def convert_dtype(df):
        names = df.columns.to_list()
        dtypes = df.dtypes.to_list()
        for i, col in enumerate(df.columns.to_list()):
            if dtypes[i] == pd.Int64Dtype():
                df.loc[:, col] = df.loc[:, col].astype(float)
        return df
    static = convert_dtype(static)

    [(vital_train, vital_dev, vital_test), (Y_train, Y_dev, Y_test), (static_train, static_dev, static_test)] = [
        [df[df.index.get_level_values(2).isin(s)] for s in (train_stay, dev_stay, test_stay)] \
        for df in (vital_final, intervention, static)]

    if args.exit_point == 'All':
        print('Exit point is after all steps, including train-val-test splitting, saving results...')
        vital_train.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='vital_train')
        vital_dev.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='vital_dev')
        vital_test.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='vital_test')
        Y_train.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='inv_train')
        Y_dev.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='inv_dev')
        Y_test.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='inv_test')
        static_train.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='static_train')
        static_dev.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='static_dev')
        static_test.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='static_test')
    return


def extract_eicu(args):

    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    client = bigquery.Client(project=args.project_id)
    ID_COLS = ['patientunitstayid']
    # minutes to hour
    to_hours = lambda x: int(x // (60 * args.time_window))
    tw_in_min = 60 * args.time_window
    # get patient group
    patient = get_patient_group_eicu(args, client)
    print("Patient icu info query done, start querying variables in Dynamic table")
    patient['unitadmitoffset'] = 0
    young_age = [str(i) for i in range(args.age_min)]
    patient = patient.loc[~patient.loc[:, 'age'].isin(young_age)]
    icuids_to_keep = patient['patientunitstayid']
    icuids_to_keep = set([str(s) for s in icuids_to_keep])
    patient.set_index('patientunitstayid', inplace=True)
    patient['max_hours'] = (patient['unitdischargeoffset'] - patient['unitadmitoffset']).apply(to_hours)
    missing_hours_fill = range_unnest(patient, 'max_hours', out_col_name='hours_in', reset_index=True)
    missing_hours_fill['tmp'] = np.NaN
    fill_df = patient.reset_index()[ID_COLS].join(missing_hours_fill.set_index('patientunitstayid'),
                                                  on='patientunitstayid')
    fill_df.set_index(ID_COLS + ['hours_in'], inplace=True)

    # blood gas
    bg = query_bg_eicu(client, icuids_to_keep)
    bg = fill_query(bg, fill_df, tw_in_min)

    # lab
    lab = query_lab_eicu(client, icuids_to_keep)
    lab = fill_query(lab, fill_df, tw_in_min)

    # vital
    vital = query_vital_eicu(client, icuids_to_keep)
    vital.drop('entryoffset', axis=1, inplace=True)
    vital = fill_query(vital, fill_df, tw_in_min)

    # microlab
    microlab = query_microlab_eicu(client, icuids_to_keep)
    microlab['hours_in'] = microlab['culturetakenoffset'].floordiv(60)
    microlab.drop(columns=['culturetakenoffset'], inplace=True)
    microlab.reset_index(inplace=True)
    microlab = microlab.groupby(ID_COLS + ['hours_in']).agg(['last'])
    microlab = microlab.reindex(fill_df.index)

    # gcs
    gcs = query_gcs_eicu(client, icuids_to_keep)
    gcs = fill_query(gcs, fill_df, tw_in_min)

    # uo
    uo = query_uo_eicu(client, icuids_to_keep)
    uo = fill_query(uo, fill_df, tw_in_min)

    # weight
    weight = query_weight_eicu(client, icuids_to_keep)
    weight = fill_query(weight, fill_df, tw_in_min)

    # cvp
    cvp = query_cvp_eicu(client, icuids_to_keep)
    cvp.loc[:, 'cvp'] = cvp.loc[:, 'cvp'].astype(float) # other wise it converts to pandas.Float64Dtype which is problematic
    cvp = fill_query(cvp, fill_df, tw_in_min, time='observationoffset')

    # concat all
    vital = bg.join([lab, vital, gcs, uo, weight, cvp, microlab])

    del bg, lab, gcs, uo, weight, cvp, microlab

    # prepare some other lab variables
    # not perfect it affects percentage calculation
    labmakeup = query_labmakeup_eicu(client, icuids_to_keep)
    labmakeup = fill_query(labmakeup, fill_df, tw_in_min)
    # tidal volume
    tidal_vol_obs = query_tidalvol_eicu(client, icuids_to_keep)
    tidal_vol_obs = fill_query(tidal_vol_obs, fill_df, tw_in_min)
    vital = vital.join([labmakeup, tidal_vol_obs])
    del labmakeup, tidal_vol_obs

    # fix invasive and non-invasive blood pressure measurement
    idx = pd.IndexSlice
    vital.loc[:, idx[:, 'count']] = vital.loc[:, idx[:, 'count']].fillna(0)

    original = vital.loc[:, idx['ibp_systolic', ['mean', 'count']]].copy(deep=True)
    makeups = vital.loc[:, idx['nibp_systolic', ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    vital.loc[:, idx['ibp_systolic', ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values

    original = vital.loc[:, idx['ibp_diastolic', ['mean', 'count']]].copy(deep=True)
    makeups = vital.loc[:, idx['nibp_diastolic', ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    vital.loc[:, idx['ibp_diastolic', ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values

    original = vital.loc[:, idx['ibp_mean', ['mean', 'count']]].copy(deep=True)
    makeups = vital.loc[:, idx['nibp_mean', ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    vital.loc[:, idx['ibp_mean', ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values

    # drop 'basedeficit' and fix datatype in a fwe columns
    vital.drop('basedeficit', axis=1, level=0, inplace=True)
    vital.drop('index', axis=1, level=0, inplace=True)
    vital = pd.get_dummies(vital)
    # screen and positive culture needs impute, they are last columns but with float data type
    vital[('positive', 'mask')] = (~vital[('positive', 'last')].isnull()).astype(float)
    vital[('screen', 'mask')] = (~vital[('screen', 'last')].isnull()).astype(float)
    vital[('has_sensitivity', 'mask')] = (~vital[('has_sensitivity', 'last')].isnull()).astype(float)

    # make empty columns
    with open("./json_files/eicu_empty_columns.json") as f:
        columns_to_make = json.load(f)
    for col in columns_to_make:
        vital[(col, 'mean')] = fill_df.values
        vital[(col, 'count')] = 0

    # other empty columns in culturesize since eicu culturesite is less complicated compared with mimic
    with open("./json_files/eicu_empty_culture.json") as f:
        empty_culture = json.load(f)
    for col in empty_culture:
        vital[col] = 0

    # organize columns
    with open("./json_files/eicu_col_order.json") as f:
        col = json.load(f)

    # generate final col
    breakpoint1 = col.index('positive')
    breakpoint2 = col.index("('culturesite', 'last')_culturesite0")
    col_ready = []
    for i in range(breakpoint1):
        col_ready.append((col[i], 'mean'))
        col_ready.append((col[i], 'count'))
    for i in range(breakpoint1, breakpoint2):
        col_ready.append((col[i], 'last'))
        col_ready.append((col[i], 'mask'))
    for i in range(breakpoint2, len(col)):
        col_ready.append(col[i])

    vital = vital[col_ready]
    print('Start querying variables in the Intervention table')

    # Intervention table
    # ventilation
    vent= query_vent_eicu(client, icuids_to_keep, tw_in_min)
    vent_data = process_inv(vent, 'vent')
    ids_with = vent_data['patientunitstayid']
    ids_with = set(map(int, ids_with))
    ids_all = set(map(int, icuids_to_keep))
    ids_without = (ids_all - ids_with)

    # patient.set_index('patientunitstayid', inplace=True)
    icustay_timediff_tmp = patient['unitdischargeoffset'] - patient['unitadmitoffset']
    icustay_timediff = pd.Series([timediff // tw_in_min
                                  for timediff in icustay_timediff_tmp], index=patient.index.values)
    # Create a new fake dataframe with blanks on all vent entries
    out_data = fill_df.copy(deep=True)
    out_data = out_data.reset_index()
    out_data = out_data.set_index('patientunitstayid')
    out_data = out_data.iloc[out_data.index.isin(ids_without)]
    out_data = out_data.reset_index()
    out_data = out_data[['patientunitstayid']]
    out_data['max_hours'] = out_data['patientunitstayid'].map(icustay_timediff)

    # Create all 0 column for vent
    out_data = out_data.groupby('patientunitstayid')
    out_data = out_data.apply(add_blank_indicators_e)
    out_data.rename(columns={'on': 'vent'}, inplace=True)

    out_data = out_data.reset_index()
    intervention = pd.concat([vent_data[['patientunitstayid', 'hours_in', 'vent']],
                              out_data[['patientunitstayid', 'hours_in', 'vent']]],
                             axis=0)

    # vasoactive drugs
    column_names = ['dopamine', 'epinephrine', 'norepinephrine', 'phenylephrine', 'vasopressin', 'dobutamine',
                    'milrinone', 'heparin']

    for c in column_names:
        med = query_med_eicu(client, icuids_to_keep, c, tw_in_min)
        # 'epinephrine',  'dopamine', 'norepinephrine', 'phenylephrine', \
        #    'vasopressin', 'dobutamine', 'milrinone',  'heparin',
        med = process_inv(med, c)
        intervention = intervention.merge(
            med[['patientunitstayid', 'hours_in', c]],
            on=['patientunitstayid', 'hours_in'],
            how='left'
        )

    # antibiotics
    anti = query_anti_eicu(client, icuids_to_keep, tw_in_min)
    anti = process_inv(anti, 'antib')
    intervention = intervention.merge(
        anti[['patientunitstayid', 'hours_in', 'antib']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    # crrt
    crrt = query_crrt_eicu(client, icuids_to_keep, tw_in_min)
    crrt = process_inv(crrt, 'crrt')
    intervention = intervention.merge(
        crrt[['patientunitstayid', 'hours_in', 'crrt']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    # rbc transfusion
    rbc = query_rbc_trans_eicu(client, icuids_to_keep, tw_in_min)
    rbc = process_inv(rbc, 'rbc')
    intervention = intervention.merge(
        rbc[['patientunitstayid', 'hours_in', 'rbc']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    # ffp transfusion
    ffp = query_ffp_trans_eicu(client, icuids_to_keep, tw_in_min)
    ffp = process_inv(ffp, 'ffp')
    intervention = intervention.merge(
        ffp[['patientunitstayid', 'hours_in', 'ffp']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    # platelets transfusion
    platelets = query_pll_trans_eicu(client, icuids_to_keep, tw_in_min)
    platelets = process_inv(platelets, 'platelets')
    intervention = intervention.merge(
        platelets[['patientunitstayid', 'hours_in', 'platelets']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    #colloid
    colloid = query_colloid_eicu(client, icuids_to_keep, tw_in_min)
    colloid = process_inv(colloid, 'colloid')
    intervention = intervention.merge(
        colloid[['patientunitstayid', 'hours_in', 'colloid']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    #crystalloid
    crystalloid = query_crystalloid_eicu(client, icuids_to_keep, tw_in_min)
    crystalloid = process_inv(crystalloid, 'crystalloid')
    intervention = intervention.merge(
        crystalloid[['patientunitstayid', 'hours_in', 'crystalloid']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    # for each column, astype to int and fill na with 0
    intervention = intervention.fillna(0)
    for i in range(3, 18):
        intervention.iloc[:, i] = intervention.iloc[:, i].astype(int)

    intervention.set_index(ID_COLS + ['hours_in'], inplace=True)
    intervention.sort_index(level=['patientunitstayid', 'hours_in'], inplace=True)

    # reorder intervention columns
    with open("./json_files/eicu_inv_col_order.json") as f:
        new_col = json.load(f)
    intervention = intervention.loc[:, new_col]
    print('Start querying variables in the Static table')

    # static query
    # commo
    commo = query_comorbidity_eicu(client, icuids_to_keep)
    commo.set_index('patientunitstayid', inplace=True)
    static = patient.join(commo)
    static_col = static.columns.tolist()
    static_col.remove('hospitalid')
    static_col.append('hospitalid')
    static = static[static_col]

    if args.exit_point == 'Raw':
        print('Exit point is after querying raw records, saving results...')
        intervention.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_inv.h5'), key='eicu_inv')
        static.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_static.h5'), key='eicu_static')
        vital.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_vital.h5'), key='eicu_vital')
        return

    total_cols = vital.columns.tolist()
    mean_col = [i for i in total_cols if 'mean' in i]
    X_mean = vital.loc[:, mean_col]

    if not args.no_removal:
        print('Performing outlier removal')
        with open("./json_files/eicu_outlier_high.json") as f:
            range_dict_high = json.load(f)
        with open("./json_files/eicu_outlier_low.json") as f:
            range_dict_low = json.load(f)
        for var_to_remove in range_dict_high:
            remove_outliers_h(vital, X_mean, var_to_remove, range_dict_high[var_to_remove])
        for var_to_remove in range_dict_low:
            remove_outliers_l(vital, X_mean, var_to_remove, range_dict_low[var_to_remove])
    else:
        print('Skipped outlier removal')
    del X_mean

    if args.exit_point == 'Outlier_removal':
        print('Exit point is after removing outliers, saving results...')
        intervention.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_inv.h5'), key='eicu_inv')
        static.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_static.h5'), key='eicu_static')
        vital.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_vital.h5'), key='eicu_vital')
        return

    # read_mimic col means col stds
    print('Start normalization and data imputation ')
    # normalize
    count_col = [i for i in total_cols if 'count' in i]
    # # fix fio2 column by x100
    # vital.loc[:, [('fio2', 'mean')]] = vital.loc[:, [('fio2', 'mean')]] * 100
    # col_means, col_stds = vital.loc[:, mean_col].mean(axis=0), vital.loc[:, mean_col].std(axis=0)
    # first use mimic mean to normorlize
    if args.norm_eicu == 'MIMIC':
        mimic_mean_std = pd.read_hdf(os.path.join(args.output_dir, 'MIMIC_mean_std_stats.h5'), key='MIMIC_mean_std')
        col_means, col_stds = mimic_mean_std.loc[:, 'mean'], mimic_mean_std.loc[:, 'std']
        col_means.index = mean_col
        col_stds.index = mean_col
    else:
        col_means, col_stds = vital.loc[:, mean_col].mean(axis=0), vital.loc[:, mean_col].std(axis=0)
    vital.loc[:, mean_col] = (vital.loc[:, mean_col] - col_means) / col_stds
    icustay_means = vital.loc[:, mean_col].groupby(ID_COLS).mean()
    # impute
    vital.loc[:, mean_col] = vital.loc[:, mean_col].groupby(ID_COLS).fillna(method='ffill').groupby(ID_COLS).fillna(
        icustay_means).fillna(0)
    # 0 or 1
    vital.loc[:, count_col] = (vital.loc[:, count_col] > 0).astype(float)
    # at this satge only 3 last columns has nan values
    vital = vital.fillna(0)
    vital[('screen', 'last')] = vital[('screen', 'last')].astype('uint8')
    vital[('positive', 'last')] = vital[('positive', 'last')].astype('uint8')
    vital[('has_sensitivity', 'last')] = vital[('has_sensitivity', 'last')].astype('uint8')

    if args.exit_point == 'Impute':
        print('Exit point is after data imputation, saving results...')
        intervention.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_inv.h5'), key='eicu_inv')
        static.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_static.h5'), key='eicu_static')
        vital.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_vital.h5'), key='eicu_vital')
        return

    # split data
    stays_v = set(vital.index.get_level_values(0).values)
    stays_static = set(static.index.get_level_values(0).values)
    stays_int = set(intervention.index.get_level_values(0).values)
    assert stays_v == stays_static, "Stay ID pools differ!"
    assert stays_v == stays_int, "Stay ID pools differ!"
    train_frac, dev_frac, test_frac = 0.7, 0.1, 0.2
    SEED = 41
    np.random.seed(SEED)
    subjects, N = np.random.permutation(list(stays_v)), len(stays_v)
    N_train, N_dev, N_test = int(train_frac * N), int(dev_frac * N), int(test_frac * N)
    train_stay = list(stays_v)[:N_train]
    dev_stay = list(stays_v)[N_train:N_train + N_dev]
    test_stay = list(stays_v)[N_train + N_dev:]
    def convert_dtype(df):
        names = df.columns.to_list()
        dtypes = df.dtypes.to_list()
        for i, col in enumerate(df.columns.to_list()):
            if dtypes[i] == pd.Int64Dtype():
                df.loc[:, col] = df.loc[:, col].astype(float)
        return df
    static = convert_dtype(static)

    [(vital_train, vital_dev, vital_test), (Y_train, Y_dev, Y_test), (static_train, static_dev, static_test)] = [
        [df[df.index.get_level_values(0).isin(s)] for s in (train_stay, dev_stay, test_stay)] \
        for df in (vital, intervention, static)]

    if args.exit_point == 'All':
        print('Exit point is after all steps, including train-val-test splitting, saving results...')
        vital_train.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='vital_train')
        vital_dev.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='vital_dev')
        vital_test.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='vital_test')
        Y_train.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='inv_train')
        Y_dev.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='inv_dev')
        Y_test.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='inv_test')
        static_train.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='static_train')
        static_dev.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='static_dev')
        static_test.to_hdf(os.path.join(args.output_dir, 'eICU_split.hdf5'), key='static_test')
    return

