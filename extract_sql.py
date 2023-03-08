'''
SQL script to extract data from the database
'''
import pandas as pd


def gcp2df(client, sql, job_config=None):
    que = client.query(sql, job_config)
    results = que.result()
    return results.to_dataframe()


def get_group_id(args, client):
    if args.patient_group == 'sepsis_3':
        query = \
            """
            SELECT  stay_id
            FROM physionet-data.mimic_derived.sepsis3
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['stay_id']])
    elif args.patient_group == 'ARF':

        query = \
            """
            SELECT DISTINCT stay_id
            FROM physionet-data.mimic_icu.chartevents
            WHERE itemid = 224700 or  itemid = 220339

            UNION ALL

            SELECT i.stay_id
            FROM physionet-data.mimic_hosp.labevents l
            LEFT JOIN physionet-data.mimic_icu.icustays i on l.subject_id = i.subject_id 
            WHERE l.itemid = 50819 
            AND l.charttime between i.intime and i.outtime 

            UNION ALL 

            SELECT DISTINCT v.stay_id 
            FROM physionet-data.mimic_derived.ventilation v
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['stay_id']])
    elif args.patient_group == 'Shock':
        query = \
            """
            SELECT DISTINCT stay_id
            FROM physionet-data.mimic_derived.vasoactive_agent
            WHERE norepinephrine is not null 
            OR epinephrine is not null 
            OR dopamine is not null 
            OR vasopressin is not null 
            OR phenylephrine  is not null 
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['stay_id']])
    elif args.patient_group == 'CHF':
        query = \
            """
            SELECT DISTINCT i.stay_id
            FROM physionet-data.mimic_derived.charlson c
            LEFT JOIN physionet-data.mimic_icu.icustays i on c.hadm_id = i.hadm_id 
            WHERE c.congestive_heart_failure = 1 and i.stay_id is not null
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['stay_id']])
    elif args.patient_group == 'COPD':
        query = \
            """
            SELECT DISTINCT i.stay_id
            FROM physionet-data.mimic_derived.charlson c
            LEFT JOIN physionet-data.mimic_icu.icustays i on c.hadm_id = i.hadm_id 
            WHERE c.chronic_pulmonary_disease = 1 and i.stay_id is not null
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['stay_id']])
    elif args.custom_id == True:
        custom_ids = pd.read_csv(args.customid_dir)
        group_stay_ids = set([str(s) for s in custom_ids['stay_id']])

    return group_stay_ids


def get_patient_group(args, client):
    # define our patient cohort by age, icu stay time
    if args.patient_group != 'Generic':
        query = \
            """
            SELECT DISTINCT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.gender,
                i.admission_age as age,
                i.ethnicity,
                i.hospital_expire_flag,
                i.hospstay_seq,
                i.los_icu,
                i.admittime,
                i.dischtime,
                i.icu_intime,
                i.icu_outtime,
                a.admission_type,
                a.insurance,
                a.deathtime,
                a.discharge_location,
                CASE when a.deathtime between i.icu_intime and i.icu_outtime THEN 1 ELSE 0 END AS mort_icu,
                CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
                COALESCE(f.readmission_30, 0) AS readmission_30
            FROM physionet-data.mimic_derived.icustay_detail i
                INNER JOIN physionet-data.mimic_core.admissions a ON i.hadm_id = a.hadm_id
                INNER JOIN physionet-data.mimic_icu.icustays s ON i.stay_id = s.stay_id
                LEFT OUTER JOIN (SELECT d.stay_id, 1 as readmission_30
                            FROM physionet-data.mimic_icu.icustays c, physionet-data.mimic_icu.icustays d
                            WHERE c.subject_id=d.subject_id
                            AND c.stay_id > d.stay_id
                            AND c.intime - d.outtime <= INTERVAL 30 DAY
                            AND c.outtime = (SELECT MIN(e.outtime) from physionet-data.mimic_icu.icustays e 
                                            WHERE e.subject_id=c.subject_id
                                            AND e.intime>d.outtime) ) f
                            ON i.stay_id=f.stay_id
            WHERE i.hadm_id is not null and i.stay_id is not null and i.stay_id in ({group_icuids})
                and i.hospstay_seq = 1
                and i.icustay_seq = 1
                and i.admission_age >= {min_age}
                and (i.icu_outtime >= (i.icu_intime + INTERVAL {min_los} Hour))
                and (i.icu_outtime <= (i.icu_intime + INTERVAL {max_los} Hour))
            ORDER BY subject_id
            ;
            """.format(group_icuids=','.join(get_group_id(args, client)), min_age=args.age_min, min_los=args.los_min,
                       max_los=args.los_max)
        patient = gcp2df(client, query)
    else:
        query = \
            """
            SELECT DISTINCT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.gender,
                i.admission_age as age,
                i.ethnicity,
                i.hospital_expire_flag,
                i.hospstay_seq,
                i.los_icu,
                i.admittime,
                i.dischtime,
                i.icu_intime,
                i.icu_outtime,
                a.admission_type,
                a.insurance,
                a.deathtime,
                a.discharge_location,
                CASE when a.deathtime between i.icu_intime and i.icu_outtime THEN 1 ELSE 0 END AS mort_icu,
                CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
                COALESCE(f.readmission_30, 0) AS readmission_30
            FROM physionet-data.mimic_derived.icustay_detail i
                INNER JOIN physionet-data.mimic_core.admissions a ON i.hadm_id = a.hadm_id
                INNER JOIN physionet-data.mimic_icu.icustays s ON i.stay_id = s.stay_id
                LEFT OUTER JOIN (SELECT d.stay_id, 1 as readmission_30
                            FROM physionet-data.mimic_icu.icustays c, physionet-data.mimic_icu.icustays d
                            WHERE c.subject_id=d.subject_id
                            AND c.stay_id > d.stay_id
                            AND c.intime - d.outtime <= INTERVAL 30 DAY
                            AND c.outtime = (SELECT MIN(e.outtime) from physionet-data.mimic_icu.icustays e 
                                            WHERE e.subject_id=c.subject_id
                                            AND e.intime>d.outtime) ) f
                            ON i.stay_id=f.stay_id
            WHERE i.hadm_id is not null and i.stay_id is not null
                and i.hospstay_seq = 1
                and i.icustay_seq = 1
                and i.admission_age >= {min_age}
                and (i.icu_outtime >= (i.icu_intime + INTERVAL {min_los} Hour))
                and (i.icu_outtime <= (i.icu_intime + INTERVAL {max_los} Hour))
            ORDER BY subject_id
            ;
            """.format(min_age=args.age_min, min_los=args.los_min, max_los=args.los_max)

        patient = gcp2df(client, query)

    return patient


def query_bg_mimic(client, subject_to_keep):
    query = """
    SELECT b.*, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.bg b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON b.subject_id = i.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))

    bg = gcp2df(client, query)

    return bg


def query_vitals_mimic(client, icuids_to_keep):
    query = """
        With vitalsign as 
        (
          select
            ce.subject_id
          , ce.stay_id
          , ce.charttime
          , AVG(case when itemid in (220045) and valuenum > 0 and valuenum < 9999 then valuenum else null end) as heart_rate
          , AVG(case when itemid in (220179,220050) and valuenum > 0 and valuenum < 9999 then valuenum else null end) as sbp
          , AVG(case when itemid in (220180,220051) and valuenum > 0 and valuenum < 9999 then valuenum else null end) as dbp
          , AVG(case when itemid in (220052,220181,225312) and valuenum > 0 and valuenum < 9999 then valuenum else null end) as mbp
          , AVG(case when itemid = 220179 and valuenum > 0 and valuenum < 9999 then valuenum else null end) as sbp_ni
          , AVG(case when itemid = 220180 and valuenum > 0 and valuenum < 9999 then valuenum else null end) as dbp_ni
          , AVG(case when itemid = 220181 and valuenum > 0 and valuenum < 9999 then valuenum else null end) as mbp_ni
          , AVG(case when itemid in (220210,224690) and valuenum > 0 and valuenum < 9999 then valuenum else null end) as resp_rate
          , ROUND(
              AVG(case when itemid in (223761) and valuenum > 70 and valuenum < 120 then (valuenum-32)/1.8 -- converted to degC in valuenum call
                      when itemid in (223762) and valuenum > 10 and valuenum < 50  then valuenum else null end)
            , 2) as temperature
          , MAX(CASE WHEN itemid = 224642 THEN value ELSE NULL END) AS temperature_site
          , AVG(case when itemid in (220277) and valuenum > 0 and valuenum <= 100 then valuenum else null end) as spo2
          , AVG(case when itemid in (225664,220621,226537) and valuenum > 0 then valuenum else null end) as glucose
          FROM  physionet-data.mimic_icu.chartevents ce
          where ce.stay_id IS NOT NULL
          and ce.itemid in
          (
            220045, -- Heart Rate
            225309, -- ART BP Systolic
            225310, -- ART BP Diastolic
            225312, -- ART BP Mean
            220050, -- Arterial Blood Pressure systolic
            220051, -- Arterial Blood Pressure diastolic
            220052, -- Arterial Blood Pressure mean
            220179, -- Non Invasive Blood Pressure systolic
            220180, -- Non Invasive Blood Pressure diastolic
            220181, -- Non Invasive Blood Pressure mean
            220210, -- Respiratory Rate
            224690, -- Respiratory Rate (Total)
            220277, -- SPO2, peripheral
            -- GLUCOSE, both lab and fingerstick
            225664, -- Glucose finger stick
            220621, -- Glucose (serum)
            226537, -- Glucose (whole blood)
            -- TEMPERATURE
            223762, -- "Temperature Celsius"
            223761,  -- "Temperature Fahrenheit"
            224642 -- Temperature Site
            -- 226329 -- Blood Temperature CCO (C)
          )
          group by ce.subject_id, ce.stay_id, ce.charttime
        )

        SELECT b.*, i.hadm_id, i.icu_intime
        FROM vitalsign b 
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON b.stay_id = i.stay_id
        where b.stay_id in ({icuids})
        and b.charttime between i.icu_intime and i.icu_outtime
        """.format(icuids=','.join(icuids_to_keep))
    vitalsign = gcp2df(client, query)

    return vitalsign


def query_blood_diff_mimic(client, subject_to_keep):
    query = """
        SELECT b.*, i.stay_id, i.icu_intime
        FROM physionet-data.mimic_derived.blood_differential b
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
        where b.subject_id in ({icuids})
        and b.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(subject_to_keep))
    blood_diff = gcp2df(client, query)

    return blood_diff


def query_cardiac_marker_mimic(client, subject_to_keep):
    query = """
        SELECT b.*, i.stay_id, i.icu_intime
        FROM physionet-data.mimic_derived.cardiac_marker b
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON b.subject_id = i.subject_id
        where b.subject_id in ({icuids})
        and b.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(subject_to_keep))
    cardiac_marker = gcp2df(client, query)
    return cardiac_marker


def query_chemistry_mimic(client, subject_to_keep):
    query = """
        With chem as 
        (
          SELECT 
            MAX(subject_id) AS subject_id
            , MAX(hadm_id) AS hadm_id
            , MAX(charttime) AS charttime
            , le.specimen_id
            -- convert from itemid into a meaningful column
            , MAX(CASE WHEN itemid = 50862 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS albumin
            , MAX(CASE WHEN itemid = 50930 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS globulin
            , MAX(CASE WHEN itemid = 50976 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS total_protein
            , MAX(CASE WHEN itemid = 50868 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS aniongap
            , MAX(CASE WHEN itemid = 50882 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS bicarbonate
            , MAX(CASE WHEN itemid = 51006 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS bun
            , MAX(CASE WHEN itemid = 50893 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS calcium
            , MAX(CASE WHEN itemid = 50902 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS chloride
            , MAX(CASE WHEN itemid = 50912 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS creatinine
            , MAX(CASE WHEN itemid = 50931 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS glucose
            , MAX(CASE WHEN itemid = 50983 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS sodium
            , MAX(CASE WHEN itemid = 50971 AND valuenum <=  9999 THEN valuenum ELSE NULL END) AS potassium
            FROM physionet-data.mimic_hosp.labevents le
            WHERE le.itemid IN(
            -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
            50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
            50930, -- Globulin
            50976, -- Total protein
            50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
            -- 52456, -- Anion gap, point of care test
            50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
            50893, -- Calcium
            50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
            -- 52502, Creatinine, point of care
            50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
            50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
            -- 52525, Glucose, point of care
            50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
            -- 52566, -- Potassium, point of care
            50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
            -- 52579, -- Sodium, point of care
            51006  -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
            -- 52603, Urea, point of care 
            )
            AND valuenum IS NOT NULL
          -- lab values cannot be 0 and cannot be negative
          -- .. except anion gap.
            AND (valuenum > 0 OR itemid = 50868)
          GROUP BY le.specimen_id
        )
        SELECT b.*, i.stay_id, i.icu_intime
        FROM chem b
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
        where b.subject_id in ({icuids})
        and b.charttime between i.icu_intime and i.icu_outtime
        """.format(icuids=','.join(subject_to_keep))
    chemistry = gcp2df(client, query)
    return chemistry


def query_coagulation_mimic(client, subject_to_keep):
    query = """
        SELECT b.*, i.stay_id, i.icu_intime
        FROM physionet-data.mimic_derived.coagulation b
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
        where b.subject_id in ({icuids})
        and b.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(subject_to_keep))
    coagulation = gcp2df(client, query)
    return coagulation


def query_cbc_mimic(client, subject_to_keep):
    query = """
        SELECT b.*, i.stay_id, i.icu_intime
        FROM physionet-data.mimic_derived.complete_blood_count b
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON b.subject_id = i.subject_id
        where b.subject_id in ({icuids})
        and b.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(subject_to_keep))
    cbc = gcp2df(client, query)
    return cbc


def query_culture_mimic(client, subject_to_keep):
    query = """
        SELECT b.subject_id, b.charttime, b.specimen, b.screen, b.positive_culture, b.has_sensitivity, 
        i.hadm_id, i.stay_id, i.icu_intime
        FROM physionet-data.mimic_derived.culture b
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
        where b.subject_id in ({icuids})
        and b.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(subject_to_keep))
    culture = gcp2df(client, query)
    return culture


def query_enzyme_mimic(client, subject_to_keep):
    query = """
        SELECT b.*, i.stay_id, i.icu_intime
        FROM physionet-data.mimic_derived.enzyme b
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
        where b.subject_id in ({icuids})
        and b.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(subject_to_keep))
    enzyme = gcp2df(client, query)
    return enzyme


def query_gcs_mimic(client, icuids_to_keep):
    query = """
        SELECT g.subject_id, g.stay_id, g.charttime, g.gcs, i.hadm_id, i.icu_intime
        FROM physionet-data.mimic_derived.gcs g
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = g.stay_id
        where g.stay_id in ({icuids})
        and g.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(icuids_to_keep))

    gcs = gcp2df(client, query)
    return gcs


def query_inflammation_mimic(client, subject_to_keep):
    # query inflammation
    query = """
        SELECT g.subject_id, g.hadm_id, g.charttime, g.crp, i.stay_id, i.icu_intime
        FROM physionet-data.mimic_derived.inflammation g 
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = g.subject_id
        where g.subject_id in ({icuids})
        and g.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(subject_to_keep))
    inflammation = gcp2df(client, query)
    return inflammation


def query_uo_mimic(client, icuids_to_keep):
    query = """
        SELECT g.stay_id, g.charttime, g.weight, g.uo, i.icu_intime, i.subject_id, i.hadm_id
        FROM physionet-data.mimic_derived.urine_output_rate g 
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = g.stay_id
        where g.stay_id in ({icuids})
        and g.charttime between i.icu_intime and i.icu_outtime

        """.format(icuids=','.join(icuids_to_keep))
    uo = gcp2df(client, query)
    return uo


def query_chart_lab_mimic(client, icuids_to_keep, chart_items, lab_items):
    query = \
        """
        SELECT c.subject_id, i.hadm_id, c.stay_id, c.charttime, c.itemid, c.value, c.valueuom
        FROM `physionet-data.mimic_derived.icustay_detail` i
        INNER JOIN `physionet-data.mimic_icu.chartevents` c ON i.stay_id = c.stay_id
        WHERE c.stay_id IN ({icuids})
            AND c.itemid IN ({chitem})
            AND c.charttime between i.icu_intime and i.icu_outtime
            AND c.valuenum is not null

        UNION ALL

        SELECT DISTINCT i.subject_id, i.hadm_id, i.stay_id, l.charttime, l.itemid, l.value, l.valueuom
        FROM `physionet-data.mimic_derived.icustay_detail` i
        INNER JOIN `physionet-data.mimic_hosp.labevents` l ON i.hadm_id = l.hadm_id
        WHERE i.stay_id  IN ({icuids})
            and l.itemid  IN ({labitem})
            and l.charttime between i.icu_intime and i.icu_outtime
            and l.valuenum > 0
        ;
        """.format(icuids=','.join(icuids_to_keep), chitem=','.join(chart_items), labitem=','.join(lab_items))

    chart_lab = gcp2df(client, query)
    return chart_lab


def query_vent_mimic(client, icuids_to_keep):
    query = """
        select i.subject_id, i.hadm_id, v.stay_id, v.starttime, v.endtime, i.icu_intime, i.icu_outtime
        FROM physionet-data.mimic_derived.icustay_detail i
        INNER JOIN physionet-data.mimic_derived.ventilation v ON i.stay_id = v.stay_id
        where v.stay_id in ({icuids})
        and v.starttime < i.icu_outtime
        and v.endtime > i.icu_intime
        """.format(icuids=','.join(icuids_to_keep))

    vent_data = gcp2df(client, query)
    return vent_data


def query_antibiotics_mimic(client, icuids_to_keep):
    query = """
        select i.subject_id, i.hadm_id, v.stay_id, v.starttime, v.stoptime as endtime, v.antibiotic, 
        v.route, i.icu_intime, i.icu_outtime 
        FROM physionet-data.mimic_derived.icustay_detail i
        INNER JOIN physionet-data.mimic_derived.antibiotic v ON i.stay_id = v.stay_id
        where v.stay_id in ({icuids})
        and v.starttime < i.icu_outtime 
        and v.stoptime > i.icu_intime 
        ;
        """.format(icuids=','.join(icuids_to_keep))

    antibiotics = gcp2df(client, query)
    return antibiotics


def query_vasoactive_mimic(client, icuids_to_keep, vasoactive_drugs):
    query = """
            select i.subject_id, i.hadm_id, v.stay_id, v.starttime, v.endtime, i.icu_intime, i.icu_outtime, 
            FROM physionet-data.mimic_derived.icustay_detail i
            INNER JOIN physionet-data.mimic_derived.vasoactive_agent v ON i.stay_id = v.stay_id
            where v.stay_id in ({icuids})
            and v.starttime  < i.icu_outtime
            and v.endtime > i.icu_intime 
            and v.{drug_name} is not null
            ;
            """.format(icuids=','.join(icuids_to_keep), drug_name=vasoactive_drugs)

    # job_config = bigquery.QueryJobConfig(query_parameters=[
    #     bigquery.ScalarQueryParameter("NAME", "STRING", c)])

    new_data = gcp2df(client, query)
    return new_data


def query_heparin_mimic(client, subject_to_keep):
    query = \
        """
    SELECT he.subject_id, he.starttime, he.stoptime as endtime, i.hadm_id, i.stay_id, i.icu_intime, i.icu_outtime
    FROM physionet-data.mimic_derived.heparin he
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = he.subject_id
    WHERE he.subject_id in ({ids}) 
    AND  he.starttime < i.icu_outtime
    AND  he.stoptime > i.icu_intime 
    """.format(ids=','.join(subject_to_keep))
    heparin = gcp2df(client, query)
    return heparin


def query_crrt_mimic(client, icuids_to_keep):
    query = \
        """
    SELECT cr.stay_id, MIN(cr.charttime) as starttime, MAX(cr.charttime) as endtime, i.subject_id, 
    i.hadm_id, i.icu_intime, i.icu_outtime
    FROM physionet-data.mimic_derived.crrt cr
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = cr.stay_id
    WHERE cr.stay_id in ({ids}) 
    AND  cr.charttime BETWEEN i.icu_intime AND i.icu_outtime
    GROUP BY cr.stay_id, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime

    """.format(ids=','.join(icuids_to_keep))
    crrt = gcp2df(client, query)
    return crrt


def query_rbc_trans_mimic(client, icuids_to_keep):
    query = \
        """
        WITH rbc as 
            (SELECT amount
            , amountuom
            , stay_id
            , starttime
            , endtime
            FROM physionet-data.mimic_icu.inputevents
            WHERE (itemid in
            (
            225168,  --Packed Red Blood Cells
            226368, --OR Packed RBC Intake
            227070 --PACU Packed RBC Intake
            )
            AND amount > 0
            AND stay_id in ({ids}) 
            )
            ORDER BY stay_id, endtime)

        SELECT rbc.stay_id, rbc.starttime, rbc.endtime, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        FROM rbc
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = rbc.stay_id
        WHERE starttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        ORDER BY stay_id
        """.format(ids=','.join(icuids_to_keep))
    rbc_trans = gcp2df(client, query)
    return rbc_trans


def query_pll_trans_mimic(client, icuids_to_keep):
    query = \
        """
        WITH pll as (
            SELECT amount
            , amountuom
            , stay_id
            , starttime
            , endtime
            FROM physionet-data.mimic_icu.inputevents
            WHERE (itemid in
            (
                225170,  --Platelets
                226369, --OR Platelet Intake
                227071  --PACU Platelet Intake
            )
            AND amount > 0
            AND stay_id in ({ids}) 
            )
            ORDER BY stay_id, endtime)

        SELECT pll.stay_id, pll.starttime, pll.endtime, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        FROM pll
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = pll.stay_id
        WHERE starttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        ORDER BY stay_id
        """.format(ids=','.join(icuids_to_keep))
    platelets_trans = gcp2df(client, query)
    return platelets_trans


def query_ffp_trans_mimic(client, icuids_to_keep):
    query = \
        """
        WITH ffp as (
            SELECT amount
            , amountuom
            , stay_id
            , starttime
            , endtime
            FROM physionet-data.mimic_icu.inputevents
            WHERE (itemid in
            (
                220970,  -- Fresh Frozen Plasma
                226367,  -- OR FFP Intake
                227072  -- PACU FFP Intake
            )
            AND amount > 0
            AND stay_id in ({ids}) 
            )
            ORDER BY stay_id, endtime)

        SELECT ffp.stay_id, ffp.starttime, ffp.endtime, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        FROM ffp
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = ffp.stay_id
        WHERE starttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        ORDER BY stay_id
        """.format(ids=','.join(icuids_to_keep))
    ffp_trans = gcp2df(client, query)
    return ffp_trans


def query_colloid_mimic(client, icuids_to_keep):
    query = \
        """
        with coll as
            (
            select
                mv.stay_id
            , mv.starttime as charttime
            , mv.endtime as endtime
            -- standardize the units to millilitres
            -- also metavision has floating point precision.. but we only care down to the mL
            , round(case
                when mv.amountuom = 'L'
                    then mv.amount * 1000.0
                when mv.amountuom = 'ml'
                    then mv.amount
                else null end) as amount
            from physionet-data.mimic_icu.inputevents mv
            where mv.itemid in
            (
                220864, --  Albumin 5%  7466 132 7466
                220862, --  Albumin 25% 9851 174 9851
                225174, --  Hetastarch (Hespan) 6%  82 1 82
                225795, --  Dextran 40  38 3 38
                225796  --  Dextran 70
                -- below ITEMIDs not in use
            -- 220861 | Albumin (Human) 20%
            -- 220863 | Albumin (Human) 4%
            )
            and mv.statusdescription != 'Rewritten'
            and
            -- in MetaVision, these ITEMIDs never appear with a null rate
            -- so it is sufficient to check the rate is > 100
                (
                (mv.rateuom = 'mL/hour' and mv.rate > 100)
                OR (mv.rateuom = 'mL/min' and mv.rate > (100/60.0))
                OR (mv.rateuom = 'mL/kg/hour' and (mv.rate*mv.patientweight) > 100)
                )
            and stay_id in ({ids}) 
            )
        -- remove carevue 
        -- some colloids are charted in chartevents

        select coll.stay_id, coll.charttime as starttime, coll.endtime, coll.amount as colloid_bolus, 
        i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        from coll
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = coll.stay_id
        -- just because the rate was high enough, does *not* mean the final amount was
        WHERE charttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        --group by coll.stay_id, coll.charttime, coll.endtime
        order by stay_id, charttime 
        """.format(ids=','.join(icuids_to_keep))
    colloid_bolus = gcp2df(client, query)
    return colloid_bolus


def query_crystalloid_mimic(client, icuids_to_keep):
    query = \
        """
        with crys as
            (
            select
                mv.stay_id
            , mv.starttime as charttime
            , mv.endtime 
            -- standardize the units to millilitres
            -- also metavision has floating point precision.. but we only care down to the mL
            , round(case
                when mv.amountuom = 'L'
                    then mv.amount * 1000.0
                when mv.amountuom = 'ml'
                    then mv.amount
                else null end) as amount
            from physionet-data.mimic_icu.inputevents mv
            where mv.itemid in
            (
                -- 225943 Solution
                225158, -- NaCl 0.9%
                225828, -- LR
                225944, -- Sterile Water
                225797, -- Free Water
                225159, -- NaCl 0.45%
                -- 225161, -- NaCl 3% (Hypertonic Saline)
                225823, -- D5 1/2NS
                225825, -- D5NS
                225827, -- D5LR
                225941, -- D5 1/4NS
                226089 -- Piggyback
            )
            and mv.statusdescription != 'Rewritten'
            and
            -- in MetaVision, these ITEMIDs appear with a null rate IFF endtime=starttime + 1 minute
            -- so it is sufficient to:
            --    (1) check the rate is > 240 if it exists or
            --    (2) ensure the rate is null and amount > 240 ml
                (
                (mv.rate is not null and mv.rateuom = 'mL/hour' and mv.rate > 248)
                OR (mv.rate is not null and mv.rateuom = 'mL/min' and mv.rate > (248/60.0))
                OR (mv.rate is null and mv.amountuom = 'L' and mv.amount > 0.248)
                OR (mv.rate is null and mv.amountuom = 'ml' and mv.amount > 248)
                )
            )

        select crys.stay_id, crys.charttime as starttime, crys.endtime, crys.amount as crystalloid_bolus, 
        i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        from crys
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = crys.stay_id
        WHERE charttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        --group by coll.stay_id, coll.charttime, coll.endtime
        order by stay_id, charttime;

        """.format(ids=','.join(icuids_to_keep))
    crystalloid_bolus = gcp2df(client, query)
    return crystalloid_bolus


def query_anchor_year_mimic(client, icuids_to_keep):
    query = """
        select i.subject_id, i.hadm_id, i.stay_id, i.icu_intime, i.icu_outtime, v.anchor_year, v.anchor_year_group
        FROM physionet-data.mimic_derived.icustay_detail i
        INNER JOIN physionet-data.mimic_core.patients v ON i.subject_id = v.subject_id
        where i.stay_id in ({icuids})
        ;
        """.format(icuids=','.join(icuids_to_keep))
    anchor_year = gcp2df(client, query)
    return anchor_year


def query_comorbidity_mimic(client, icuids_to_keep):
    query = """
        select c.subject_id, c.hadm_id, i.stay_id, c.myocardial_infarct, c.congestive_heart_failure, 
        c.peripheral_vascular_disease, c.cerebrovascular_disease, c.dementia, c.chronic_pulmonary_disease, 
        c.rheumatic_disease, c.peptic_ulcer_disease, c.mild_liver_disease, c.diabetes_without_cc, 
        c.diabetes_with_cc, c.paraplegia, c.renal_disease, c.malignant_cancer, c.severe_liver_disease, 
        c.metastatic_solid_tumor, c.aids
        FROM physionet-data.mimic_derived.charlson c
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.hadm_id = c.hadm_id
        where i.stay_id in ({icuids})
        """.format(icuids=','.join(icuids_to_keep))
    comorbidity = gcp2df(client, query)
    return comorbidity


def get_group_id_eicu(args, client):
    if args.patient_group == 'sepsis_3':
        sepsis3_ids = pd.read_csv('./resources/eicu_sepsis_3_id.csv')
        group_stay_ids = set([str(s) for s in sepsis3_ids['patientunitstayid']])
    elif args.patient_group == 'ARF':
        query = \
            """
            SELECT DISTINCT l.patientunitstayid
            FROM physionet-data.eicu_crd.lab l
            WHERE l.labname = 'PEEP' 
            and l.labresult >= 0 
            and l.labresult <= 30

            UNION ALL

            SELECT DISTINCT vt.patientunitstayid 
            FROM (
                SELECT Distinct i.patientunitstayid, i.priorventstartoffset, i.priorventendoffset	
                From physionet-data.eicu_crd.respiratorycare i
                WHERE i.priorventstartoffset >0 or i.priorventendoffset	>0
                Order by patientunitstayid
            ) vt
            INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = vt.patientunitstayid
            WHERE  vt.priorventstartoffset is not null 
            AND vt.priorventendoffset is not null
            AND FLOOR(LEAST(vt.priorventendoffset, i.unitdischargeoffset)/60) > FLOOR(GREATEST(vt.priorventstartoffset, 0)/60)
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['patientunitstayid']])
    elif args.patient_group == 'Shock':
        query = \
            """
            SELECT DISTINCT pm.patientunitstayid, 
            FROM physionet-data.eicu_crd_derived.pivoted_med pm
            INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = pm.patientunitstayid
            WHERE (pm.dopamine = 1 or pm.norepinephrine = 1 or pm.epinephrine = 1 or pm.vasopressin = 1 or pm. phenylephrine = 1)
            AND pm.drugorderoffset is not null 
            AND pm.drugstopoffset is not null
            AND FLOOR(LEAST(pm.drugstopoffset, i.unitdischargeoffset)/60) > FLOOR(GREATEST(pm.drugorderoffset, 0)/60)
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['patientunitstayid']])
    elif args.patient_group == 'CHF':
        query = \
            """
            SELECT DISTINCT ad.patientunitstayid,
            FROM physionet-data.eicu_crd.diagnosis ad
            WHERE SUBSTR(ad.icd9code, 1, 3) = '428'
            OR SUBSTR(ad.icd9code, 1, 6) IN ('398.91','402.01','402.11','402.91','404.01','404.03',
                                    '404.11','404.13','404.91','404.93')
            OR SUBSTR(ad.icd9code, 1, 5) BETWEEN '425.4' AND '425.9'
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['patientunitstayid']])
    elif args.patient_group == 'COPD':
        query = \
            """
            SELECT DISTINCT ad.patientunitstayid,
            FROM physionet-data.eicu_crd.diagnosis ad
            WHERE SUBSTR(ad.icd9code, 1, 3) BETWEEN '490' AND '505'
            OR SUBSTR(ad.icd9code, 1, 5) IN ('416.8','416.9','506.4','508.1','508.8')
            """
        id_df = gcp2df(client, query)
        group_stay_ids = set([str(s) for s in id_df['patientunitstayid']])
    elif args.custom_id:
        custom_ids = pd.read_csv(args.customid_dir)
        group_stay_ids = set([str(s) for s in custom_ids['stay_id']])

    return group_stay_ids


def get_patient_group_eicu(args, client):
    if args.patient_group != 'Generic':
        query = \
            """
            SELECT i.patientunitstayid, i.gender, i.age, i.ethnicity,  
                    CASE WHEN lower(i.hospitaldischargestatus) like '%alive%' THEN 0
                        WHEN lower(i.hospitaldischargestatus) like '%expired%' THEN 1
                        ELSE NULL END AS hosp_mort,
                    ROUND(i.unitdischargeoffset/60) AS icu_los_hours, i.hospitaladmitoffset, i.hospitaldischargeoffset,
                   i.unitdischargeoffset, i.hospitaladmitsource, i.unitdischargelocation, 
                   CASE WHEN lower(i.unitdischargestatus) like '%alive%' THEN 0
                        WHEN lower(i.unitdischargestatus) like '%expired%' THEN 1
                        ELSE NULL END AS icu_mort, i.hospitaldischargeyear, i.hospitalid      
            From physionet-data.eicu_crd.patient i
            WHERE ROUND(i.unitdischargeoffset/60) Between {min_los} and {max_los} 
            AND patientunitstayid in ({group_icuids})
            """.format(group_icuids=','.join(get_group_id_eicu(args, client)), min_los=args.los_min,
                       max_los=args.los_max)
        patient = gcp2df(client, query)

    else:
        query = \
            """
            SELECT i.patientunitstayid, i.gender, i.age, i.ethnicity,  
                    CASE WHEN lower(i.hospitaldischargestatus) like '%alive%' THEN 0
                        WHEN lower(i.hospitaldischargestatus) like '%expired%' THEN 1
                        ELSE NULL END AS hosp_mort,
                    ROUND(i.unitdischargeoffset/60) AS icu_los_hours, i.hospitaladmitoffset, i.hospitaldischargeoffset,
                   i.unitdischargeoffset, i.hospitaladmitsource, i.unitdischargelocation, 
                   CASE WHEN lower(i.unitdischargestatus) like '%alive%' THEN 0
                        WHEN lower(i.unitdischargestatus) like '%expired%' THEN 1
                        ELSE NULL END AS icu_mort, i.hospitaldischargeyear, i.hospitalid      
            From physionet-data.eicu_crd.patient i
            WHERE ROUND(i.unitdischargeoffset/60) Between {min_los} and {max_los} 
            """.format(min_los=args.los_min, max_los=args.los_max)
        patient = gcp2df(client, query)

    return patient


def query_bg_eicu(client, icuids_to_keep):
    query = """
    with vw0 as
    (
      select
          patientunitstayid
        , labname
        , labresultoffset
        , labresultrevisedoffset
      from physionet-data.eicu_crd.lab
      where labname in
      (
            'paO2'
          , 'paCO2'
          , 'pH'
          , 'FiO2'
          , 'anion gap'
          , 'Base Excess'
          , 'PEEP'
      )
      group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
      having count(distinct labresult)<=1
    )
    -- get the last lab to be revised
    , vw1 as
    (
      select
          lab.patientunitstayid
        , lab.labname
        , lab.labresultoffset
        , lab.labresultrevisedoffset
        , lab.labresult
        , ROW_NUMBER() OVER
            (
              PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
              ORDER BY lab.labresultrevisedoffset DESC
            ) as rn
      from physionet-data.eicu_crd.lab
      inner join vw0
        ON  lab.patientunitstayid = vw0.patientunitstayid
        AND lab.labname = vw0.labname
        AND lab.labresultoffset = vw0.labresultoffset
        AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
      WHERE
         (lab.labname = 'paO2' and lab.labresult > 0 and lab.labresult <= 9999)
      OR (lab.labname = 'paCO2' and lab.labresult > 0 and lab.labresult <= 9999)
      OR (lab.labname = 'pH' and lab.labresult >= 6.3 and lab.labresult <= 9999)
      OR (lab.labname = 'FiO2' and lab.labresult >= 0.2 and lab.labresult <= 1.0)
      -- we will fix fio2 units later
      OR (lab.labname = 'FiO2' and lab.labresult >= 20 and lab.labresult <= 100)
      OR (lab.labname = 'anion gap' and lab.labresult >= 0 and lab.labresult <= 9999)
      OR (lab.labname = 'Base Excess' and lab.labresult >= -100 and lab.labresult <= 100)
      OR (lab.labname = 'PEEP' and lab.labresult >= 0 and lab.labresult <= 9999)
    )
    select
        patientunitstayid
      , labresultoffset as chartoffset
      -- the aggregate (max()) only ever applies to 1 value due to the where clause
      , MAX(case
            when labname != 'FiO2' then null
            when labresult <= 1 then labresult*100.0
          else labresult end) as fio2
      , MAX(case when labname = 'paO2' then labresult else null end) as pao2
      , MAX(case when labname = 'paCO2' then labresult else null end) as paco2
      , MAX(case when labname = 'pH' then labresult else null end) as pH
      , MAX(case when labname = 'anion gap' then labresult else null end) as aniongap
      , MAX(case when labname = 'Base Deficit' then labresult else null end) as basedeficit
      , MAX(case when labname = 'Base Excess' then labresult else null end) as baseexcess
      , MAX(case when labname = 'PEEP' then labresult else null end) as peep
    from vw1
    where rn = 1
    and patientunitstayid in ({icuids})
    and labresultoffset >=0
    group by patientunitstayid, labresultoffset
    order by patientunitstayid, labresultoffset
    """.format(icuids=','.join(icuids_to_keep))
    bg = gcp2df(client, query)
    return bg


def query_lab_eicu(client, icuids_to_keep):
    query = """
    with vw0 as
    (
      select
          patientunitstayid
        , labname
        , labresultoffset
        , labresultrevisedoffset
      from physionet-data.eicu_crd.lab
      where labname in
      (
          'albumin'
        , 'total bilirubin'
        , 'BUN'
        , 'calcium'
        , 'chloride'
        , 'creatinine'
        , 'bedside glucose', 'glucose'
        , 'bicarbonate' -- HCO3
        , 'Total CO2'
        , 'Hct'
        , 'Hgb'
        , 'PT - INR'
        , 'PTT'
        , 'lactate'
        , 'platelets x 1000'
        , 'potassium'
        , 'sodium'
        -- cbc related 
        , 'WBC x 1000'
        , '-bands'
        , '-basos'
        , '-eos'
        , '-lymphs'
        , '-monos'
        , '-polys'
        -- Liver enzymes
        , 'ALT (SGPT)'
        , 'AST (SGOT)'
        , 'alkaline phos.'
        -- Other 
        , 'troponin - T'
        , 'CPK-MB'
        , 'total protein'
        , 'fibrinogen'
        , 'PT'
        , 'MCH'
        , 'MCHC'
        , 'MCV'
        , 'RBC'
        , 'RDW'
        , 'amylase'
        , 'CPK'
        , 'CRP'
      )
      group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
      having count(distinct labresult)<=1
    )
    -- get the last lab to be revised
    , vw1 as
    (
      select
          lab.patientunitstayid
        , lab.labname
        , lab.labresultoffset
        , lab.labresultrevisedoffset
        , lab.labresult
        , ROW_NUMBER() OVER
            (
              PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
              ORDER BY lab.labresultrevisedoffset DESC
            ) as rn
      from physionet-data.eicu_crd.lab
      inner join vw0
        ON  lab.patientunitstayid = vw0.patientunitstayid
        AND lab.labname = vw0.labname
        AND lab.labresultoffset = vw0.labresultoffset
        AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
      -- only valid lab values
      WHERE
           (lab.labname = 'albumin' and lab.labresult > 0 and lab.labresult <=9999)
        OR (lab.labname = 'total bilirubin' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'BUN' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'calcium' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'chloride' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'creatinine' and lab.labresult >0 and lab.labresult <= 9999)
        OR (lab.labname in ('bedside glucose', 'glucose') and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'bicarbonate' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'Total CO2' and lab.labresult > 0 and lab.labresult <= 9999)
        -- will convert hct unit to fraction later
        OR (lab.labname = 'Hct' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'Hgb' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'PT - INR' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'lactate' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'platelets x 1000' and lab.labresult >  0 and lab.labresult <= 9999)
        OR (lab.labname = 'potassium' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'PTT' and lab.labresult >  0 and lab.labresult <=9999)
        OR (lab.labname = 'sodium' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'WBC x 1000' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = '-bands' and lab.labresult > 0 and lab.labresult <= 100)
        OR (lab.labname = '-basos' and lab.labresult > 0)
        OR (lab.labname = '-eos' and lab.labresult > 0)
        OR (lab.labname = '-lymphs' and lab.labresult > 0)
        OR (lab.labname = '-monos' and lab.labresult > 0)
        OR (lab.labname = '-polys' and lab.labresult > 0)
        OR (lab.labname = 'ALT (SGPT)' and lab.labresult > 0)
        OR (lab.labname = 'AST (SGOT)' and lab.labresult > 0)
        OR (lab.labname = 'alkaline phos.' and lab.labresult > 0)
        OR (lab.labname = 'troponin - T' and lab.labresult > 0)
        OR (lab.labname = 'CPK-MB' and lab.labresult > 0)
        OR (lab.labname = 'total protein' and lab.labresult > 0)
        OR (lab.labname = 'fibrinogen' and lab.labresult > 0)
        OR (lab.labname = 'PT' and lab.labresult > 0)
        OR (lab.labname = 'MCH' and lab.labresult > 0)
        OR (lab.labname = 'MCHC' and lab.labresult > 0)
        OR (lab.labname = 'MCV' and lab.labresult > 0)
        OR (lab.labname = 'RBC' and lab.labresult > 0)
        OR (lab.labname = 'RDW' and lab.labresult > 0)
        OR (lab.labname = 'amylase' and lab.labresult > 0)
        OR (lab.labname = 'CPK' and lab.labresult > 0)
        OR (lab.labname = 'CRP' and lab.labresult > 0)
    )
    select
        patientunitstayid
      , labresultoffset as chartoffset
      , MAX(case when labname = 'albumin' then labresult else null end) as albumin
      , MAX(case when labname = 'total bilirubin' then labresult else null end) as bilirubin
      , MAX(case when labname = 'BUN' then labresult else null end) as BUN
      , MAX(case when labname = 'calcium' then labresult else null end) as calcium
      , MAX(case when labname = 'chloride' then labresult else null end) as chloride
      , MAX(case when labname = 'creatinine' then labresult else null end) as creatinine
      , MAX(case when labname in ('bedside glucose', 'glucose') then labresult else null end) as glucose
      , MAX(case when labname = 'bicarbonate' then labresult else null end) as bicarbonate
      , MAX(case when labname = 'Total CO2' then labresult else null end) as TotalCO2
      , MAX(case when labname = 'Hct' then labresult else null end) as hematocrit
      , MAX(case when labname = 'Hgb' then labresult else null end) as hemoglobin
      , MAX(case when labname = 'PT - INR' then labresult else null end) as INR
      , MAX(case when labname = 'lactate' then labresult else null end) as lactate
      , MAX(case when labname = 'platelets x 1000' then labresult else null end) as platelets
      , MAX(case when labname = 'potassium' then labresult else null end) as potassium
      , MAX(case when labname = 'PTT' then labresult else null end) as ptt
      , MAX(case when labname = 'sodium' then labresult else null end) as sodium
      , MAX(case when labname = 'WBC x 1000' then labresult else null end) as wbc
      , MAX(case when labname = '-bands' then labresult else null end) as bands
      , MAX(case when labname = '-basos' then labresult else null end) as basos
      , MAX(case when labname = '-eos' then labresult else null end) as eos
      , MAX(case when labname = '-lymphs' then labresult else null end) as lymphs
      , MAX(case when labname = '-monos' then labresult else null end) as monos
      , MAX(case when labname = '-polys' then labresult else null end) as polys
      , MAX(case when labname = 'ALT (SGPT)' then labresult else null end) as alt
      , MAX(case when labname = 'AST (SGOT)' then labresult else null end) as ast
      , MAX(case when labname = 'alkaline phos.' then labresult else null end) as alp
      , MAX(case when labname = 'troponin - T' then labresult else null end) as troponin_t
      , MAX(case when labname = 'CPK-MB' then labresult else null end) as cpk_mb
      , MAX(case when labname = 'total protein' then labresult else null end) as total_protein
      , MAX(case when labname = 'fibrinogen' then labresult else null end) as fibrinogen
      , MAX(case when labname = 'PT' then labresult else null end) as pt
      , MAX(case when labname = 'MCH' then labresult else null end) as mch
      , MAX(case when labname = 'MCHC' then labresult else null end) as mchc
      , MAX(case when labname = 'MCV' then labresult else null end) as mcv
      , MAX(case when labname = 'RBC' then labresult else null end) as rbc
      , MAX(case when labname = 'RDW' then labresult else null end) as rdw
      , MAX(case when labname = 'amylase' then labresult else null end) as amylase
      , MAX(case when labname = 'CPK' then labresult else null end) as cpk
      , MAX(case when labname = 'CRP' then labresult else null end) as crp
    from vw1
    where rn = 1
    and patientunitstayid in ({icuids})
    and labresultoffset >=0 
    group by patientunitstayid, labresultoffset
    order by patientunitstayid, labresultoffset
    """.format(icuids=','.join(icuids_to_keep))
    lab = gcp2df(client, query)
    return lab


def query_vital_eicu(client, icuids_to_keep):
    query = """
    with nc as
    (
    select
        patientunitstayid
      , nursingchartoffset
      , nursingchartentryoffset
      , case
          when nursingchartcelltypevallabel = 'Heart Rate'
           and nursingchartcelltypevalname = 'Heart Rate'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as heartrate
      , case
          when nursingchartcelltypevallabel = 'Respiratory Rate'
           and nursingchartcelltypevalname = 'Respiratory Rate'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as RespiratoryRate
      , case
          when nursingchartcelltypevallabel = 'O2 Saturation'
           and nursingchartcelltypevalname = 'O2 Saturation'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as o2saturation
      , case
          when nursingchartcelltypevallabel = 'Non-Invasive BP'
           and nursingchartcelltypevalname = 'Non-Invasive BP Systolic'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as nibp_systolic
      , case
          when nursingchartcelltypevallabel = 'Non-Invasive BP'
           and nursingchartcelltypevalname = 'Non-Invasive BP Diastolic'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as nibp_diastolic
      , case
          when nursingchartcelltypevallabel = 'Non-Invasive BP'
           and nursingchartcelltypevalname = 'Non-Invasive BP Mean'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as nibp_mean
      , case
          when nursingchartcelltypevallabel = 'Temperature'
           and nursingchartcelltypevalname = 'Temperature (C)'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as temperature
      --, case
      --    when nursingchartcelltypevallabel = 'Temperature'
      --     and nursingchartcelltypevalname = 'Temperature Location'
      --        then nursingchartvalue
      --    else null end
      --  as TemperatureLocation
      , case
          when nursingchartcelltypevallabel = 'Invasive BP'
           and nursingchartcelltypevalname = 'Invasive BP Systolic'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as ibp_systolic
      , case
          when nursingchartcelltypevallabel = 'Invasive BP'
           and nursingchartcelltypevalname = 'Invasive BP Diastolic'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as ibp_diastolic
      , case
          when nursingchartcelltypevallabel = 'Invasive BP'
           and nursingchartcelltypevalname = 'Invasive BP Mean'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          -- other map fields
          when nursingchartcelltypevallabel = 'MAP (mmHg)'
           and nursingchartcelltypevalname = 'Value'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          when nursingchartcelltypevallabel = 'Arterial Line MAP (mmHg)'
           and nursingchartcelltypevalname = 'Value'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as ibp_mean
      from physionet-data.eicu_crd.nursecharting
      -- speed up by only looking at a subset of charted data
      where nursingchartcelltypecat in
      (
        'Vital Signs','Scores','Other Vital Signs and Infusions'
      )
    )
    select
      patientunitstayid
    , nursingchartoffset as chartoffset
    , nursingchartentryoffset as entryoffset
    , avg(case when heartrate > 0 and heartrate <= 9999 then heartrate else null end) as heartrate
    , avg(case when RespiratoryRate >= 0 and RespiratoryRate <= 9999 then RespiratoryRate else null end) as RespiratoryRate
    , avg(case when o2saturation >= 0 and o2saturation <= 100 then o2saturation else null end) as spo2
    , avg(case when nibp_systolic > 0 and nibp_systolic <= 9999 then nibp_systolic else null end) as nibp_systolic
    , avg(case when nibp_diastolic > 0 and nibp_diastolic <= 9999 then nibp_diastolic else null end) as nibp_diastolic
    , avg(case when nibp_mean > 0 and nibp_mean <= 9999 then nibp_mean else null end) as nibp_mean
    , avg(case when temperature >= 14.2 and temperature <= 47 then temperature else null end) as temperature
    --, max(temperaturelocation) as temperaturelocation
    , avg(case when ibp_systolic > 0 and ibp_systolic <= 9999 then ibp_systolic else null end) as ibp_systolic
    , avg(case when ibp_diastolic > 0 and ibp_diastolic <= 9999 then ibp_diastolic else null end) as ibp_diastolic
    , avg(case when ibp_mean > 0 and ibp_mean <= 9999 then ibp_mean else null end) as ibp_mean
    from nc
    WHERE (heartrate IS NOT NULL
    OR RespiratoryRate IS NOT NULL
    OR o2saturation IS NOT NULL
    OR nibp_systolic IS NOT NULL
    OR nibp_diastolic IS NOT NULL
    OR nibp_mean IS NOT NULL
    OR temperature IS NOT NULL
    --OR temperaturelocation IS NOT NULL
    OR ibp_systolic IS NOT NULL
    OR ibp_diastolic IS NOT NULL
    OR ibp_mean IS NOT NULL)
    AND patientunitstayid in ({icuids})
    AND nursingchartoffset >=0 
    group by patientunitstayid, nursingchartoffset, nursingchartentryoffset
    order by patientunitstayid, nursingchartoffset, nursingchartentryoffset
    """.format(icuids=','.join(icuids_to_keep))
    vital = gcp2df(client, query)
    return vital


def query_microlab_eicu(client, icuids_to_keep):
    query = """
        SELECT ml.patientunitstayid, ml.culturetakenoffset
            , case
                when ml.culturesite = 'Blood, Venipuncture' then 'culturesite0'
                when ml.culturesite in ('Urine, Catheter Specimen', 'Urine, Voided Specimen') then 'culturesite1'
                when ml.culturesite = 'Nasopharynx' then 'culturesite2'
                when ml.culturesite = 'Stool' then 'culturesite3'
                when ml.culturesite in ('Sputum, Tracheal Specimen', 'Sputum, Expectorated') then 'culturesite4'
                when ml.culturesite = 'CSF' then 'culturesite8'
                when ml.culturesite = 'Peritoneal Fluid' then 'culturesite9'
                when ml.culturesite = 'Bronchial Lavage' then 'culturesite11'
                when ml.culturesite = 'Rectal Swab' then 'culturesite12'
                when ml.culturesite in ('Other', 'Wound, Decubitus', 'Pleural Fluid', 
                      'Bile', 'Skin', 'Wound, Surgical', 'Wound, Drainage Fluid', 'Blood, Central Line', 'Abscess')
                      then 'culturesite13'
                else null end as culturesite
            , case
                when ml.organism = 'no growth' then 0
                when ml.organism != ""  then 1
                else null end as positive
            , case 
                when ml.antibiotic != ""  then 1
                else null end as screen
            , case 
                when ml.sensitivitylevel = 'Sensitive' then 1
                when ml.sensitivitylevel = 'Resistant' then 0 
                else null end as has_sensitivity
        FROM physionet-data.eicu_crd.microlab ml
        WHERE ml.patientunitstayid in ({icuids})
        AND ml.culturetakenoffset >=0
        """.format(icuids=','.join(icuids_to_keep))
    microlab = gcp2df(client, query)
    return microlab


def query_gcs_eicu(client, icuids_to_keep):
    query = """
    SELECT gc.patientunitstayid	, gc.chartoffset, gc.gcs
    FROM physionet-data.eicu_crd_derived.pivoted_gcs gc
    WHERE gc.patientunitstayid in ({icuids})
    and gc.chartoffset >=0
    """.format(icuids=','.join(icuids_to_keep))
    gcs = gcp2df(client, query)
    return gcs


def query_uo_eicu(client, icuids_to_keep):
    query = """
    SELECT uo.patientunitstayid, uo.chartoffset, uo.urineoutput
    FROM physionet-data.eicu_crd_derived.pivoted_uo uo
    WHERE uo.patientunitstayid in ({icuids})
    and uo.chartoffset >=0
    """.format(icuids=','.join(icuids_to_keep))
    uo = gcp2df(client, query)
    return uo


def query_weight_eicu(client, icuids_to_keep):
    query = """
        SELECT wg.patientunitstayid, wg.chartoffset, wg.weight
        FROM physionet-data.eicu_crd_derived.pivoted_weight wg
        WHERE wg.patientunitstayid in ({icuids})
        and wg.chartoffset >=0
        """.format(icuids=','.join(icuids_to_keep))
    weight = gcp2df(client, query)
    return weight


def query_cvp_eicu(client, icuids_to_keep):
    query = """
    SELECT vp.patientunitstayid, vp.observationoffset, CAST(vp.cvp*0.736 AS INT64) as cvp
    FROM physionet-data.eicu_crd.vitalperiodic vp
    WHERE vp.patientunitstayid in ({icuids})
    and vp.observationoffset >=0
    """.format(icuids=','.join(icuids_to_keep))
    cvp = gcp2df(client, query)
    return cvp


def query_labmakeup_eicu(client, icuids_to_keep):
    query = """
        with vw0 as
        (
          select
              patientunitstayid
            , labname
            , labresultoffset
            , labresultrevisedoffset
          from physionet-data.eicu_crd.lab
          where labname in
          ('urinary creatinine', 'magnesium',  'phosphate', "WBC's in urine"
          )
          group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
          having count(distinct labresult)<=1
        )
        -- get the last lab to be revised
        , vw1 as
        (
          select
              lab.patientunitstayid
            , lab.labname
            , lab.labresultoffset
            , lab.labresultrevisedoffset
            , lab.labresult
            , ROW_NUMBER() OVER
                (
                  PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
                  ORDER BY lab.labresultrevisedoffset DESC
                ) as rn
          from physionet-data.eicu_crd.lab
          inner join vw0
            ON  lab.patientunitstayid = vw0.patientunitstayid
            AND lab.labname = vw0.labname
            AND lab.labresultoffset = vw0.labresultoffset
            AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
          -- only valid lab values
          WHERE
               (lab.labname = 'urinary creatinine' and lab.labresult > 0) -- based on mimic 
            OR (lab.labname = 'magnesium' and lab.labresult > 0)
            OR (lab.labname = 'phosphate' and lab.labresult > 0)
            OR (lab.labname = "WBC's in urine" and lab.labresult > 0) -- based on mimic
        )
        select
            patientunitstayid
          , labresultoffset as chartoffset
          , MAX(case when labname = 'urinary creatinine' then labresult else null end) as urine_creat
          , MAX(case when labname = 'magnesium' then labresult else null end) as magnesium
          , MAX(case when labname = 'phosphate' then labresult else null end) as phosphate
          , MAX(case when labname = "WBC's in urine" then labresult else null end) as wbc_urine
        from vw1
        where rn = 1
        and patientunitstayid in ({icuids})
        and labresultoffset >=0
        group by patientunitstayid, labresultoffset
        order by patientunitstayid, labresultoffset
        """.format(icuids=','.join(icuids_to_keep))
    labmakeup = gcp2df(client, query)
    return labmakeup


def query_tidalvol_eicu(client, icuids_to_keep):
    query = """
        SELECT rc.patientunitstayid, 
                rc.respchartoffset as chartoffset, cast(rc.respchartvalue as FLOAT64) as tidal_vol_obs
        FROM physionet-data.eicu_crd.respiratorycharting rc
        WHERE rc.respchartvaluelabel = 'Tidal Volume Observed (VT)'
        AND patientunitstayid in ({icuids})
        AND respchartoffset >=0
        """.format(icuids=','.join(icuids_to_keep))
    tidal_vol_obs = gcp2df(client, query)
    return tidal_vol_obs


def query_vent_eicu(client, icuids_to_keep, tw_in_minutes):
    # query = """
    #     SELECT v.patientunitstayid, v.chartoffset, v.ventmode
    #     FROM physionet-data.eicu_crd_derived.pivoted_ventmode v
    #     WHERE v.patientunitstayid in ({icuids})
    #     AND v.chartoffset >=0
    #     """.format(icuids=','.join(icuids_to_keep))
    query = \
        """
        with 
        ventall as
        (
            SELECT Distinct i.patientunitstayid, i.priorventstartoffset, i.priorventendoffset	
            From physionet-data.eicu_crd.respiratorycare i
            WHERE i.priorventstartoffset >0 or i.priorventendoffset	>0
            Order by patientunitstayid
        )

        SELECT vt.patientunitstayid, FLOOR(GREATEST(vt.priorventstartoffset, 0)/{tw}) as starttime, 
            FLOOR(LEAST(vt.priorventendoffset, i.unitdischargeoffset)/{tw}) as endtime,
           FLOOR((i.unitdischargeoffset - i.unitadmitoffset)/{tw}) as max_hours
        FROM ventall vt
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = vt.patientunitstayid
        WHERE  vt.priorventstartoffset is not null 
        AND vt.priorventendoffset is not null
        AND vt.patientunitstayid in ({icuids})
        """.format(icuids=','.join(icuids_to_keep), tw=tw_in_minutes)
    vent = gcp2df(client, query)
    return vent


def query_med_eicu(client, icuids_to_keep, c, tw_in_minutes):
    query = \
        """
        SELECT pm.patientunitstayid, FLOOR(GREATEST(pm.drugorderoffset, 0)/{tw}) as starttime, FLOOR(LEAST(pm.drugstopoffset, i.unitdischargeoffset)/{tw}) as endtime, 
            pm.{drug_name}, 
            FLOOR((i.unitdischargeoffset - i.unitadmitoffset)/{tw}) as max_hours
        FROM physionet-data.eicu_crd_derived.pivoted_med pm
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = pm.patientunitstayid
        WHERE pm.{drug_name} = 1 
        AND pm.patientunitstayid in ({icuids}) 
        AND pm.drugorderoffset is not null 
        AND pm.drugstopoffset is not null
        """.format(drug_name=c, icuids=','.join(icuids_to_keep), tw=tw_in_minutes)
    med = gcp2df(client, query)
    return med


def query_anti_eicu(client, icuids_to_keep, tw_in_minutes):
    query = \
        """
        SELECT md.patientunitstayid, FLOOR(GREATEST(md.drugstartoffset, 0)/{tw}) as starttime
            , FLOOR(LEAST(md.drugstopoffset, i.unitdischargeoffset)/{tw}) as endtime
            , FLOOR((i.unitdischargeoffset - i.unitadmitoffset)/{tw}) as max_hours
        FROM physionet-data.eicu_crd.medication md
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = md.patientunitstayid
        WHERE (REGEXP_CONTAINS(lower(drugname), r"^.*adoxa.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ala-tet.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*alodox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*amikacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*amikin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*amoxicill.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*amphotericin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*anidulafungin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ancef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*clavulanate.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ampicillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*augmentin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*avelox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*avidoxy.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*azactam.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*azithromycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*aztreonam.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*axetil.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bactocill.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bactrim.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bactroban.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bethkis.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*biaxin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bicillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cayston.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefazolin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cedax.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefoxitin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftazidime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefaclor.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefadroxil.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefdinir.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefditoren.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefepime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefotan.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefotetan.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefotaxime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftaroline.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefpodoxime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefpirome.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefprozil.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftibuten.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftriaxone.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefuroxime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cephalexin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cephalothin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cephapririn.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*chloramphenicol.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cipro.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ciprofloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*claforan.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*clarithromycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cleocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*clindamycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cubicin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*dicloxacillin.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*dirithromycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*doryx.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*doxycy.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*duricef.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*dynacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ery-tab.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*eryped.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*eryc.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*erythrocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*erythromycin.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*factive.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*flagyl.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*fortaz.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*furadantin.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*garamycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*gentamicin.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*kanamycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*keflex.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*kefzol.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ketek.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*levaquin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*levofloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*lincocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*linezolid.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*macrobid.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*macrodantin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*maxipime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*mefoxin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*metronidazole.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*meropenem.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*methicillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*minocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*minocycline.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*monodox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*monurol.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*morgidox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*moxatag.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*moxifloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*mupirocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*myrac.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*nafcillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*neomycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*nicazel doxy 30.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*nitrofurantoin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*norfloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*noroxin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ocudox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ofloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*omnicef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*oracea.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*oraxyl.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*oxacillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*pc pen vk.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*pce dispertab.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*panixine.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*pediazole.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*penicillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*periostat.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*pfizerpen.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*piperacillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*tazobactam.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*primsol.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*proquin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*raniclor.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*rifadin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*rifampin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*rocephin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*smz-tmp.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*septra.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*septra ds.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*septra.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*solodyn.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*spectracef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*streptomycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*sulfadiazine.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*sulfamethoxazole.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*trimethoprim.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*sulfatrim.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*sulfisoxazole.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*suprax.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*synercid.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*tazicef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*tetracycline.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*timentin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*tobramycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*trimethoprim.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*unasyn.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vancocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vancomycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vantin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vibativ.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vibra-tabs.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vibramycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*zinacef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*zithromax.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*zosyn.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*zyvox.*$")
          )
        AND md.drugordercancelled = 'No'
        AND md.patientunitstayid in ({icuids}) 
        AND md.drugstartoffset is not null 
        AND md.drugstopoffset is not null
        """.format(icuids=','.join(icuids_to_keep), tw=tw_in_minutes)

    anti = gcp2df(client, query)
    return anti


def query_crrt_eicu(client, icuids_to_keep, tw_in_minutes):
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/{tw}) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/{tw}) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/{tw}) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE REGEXP_CONTAINS(lower(cellpath), r"^.*crrt.*$")
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep), tw=tw_in_minutes)
    crrt = gcp2df(client, query)
    return crrt


def query_rbc_trans_eicu(client, icuids_to_keep, tw_in_minutes):
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/{tw}) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/{tw}) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/{tw}) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE (REGEXP_CONTAINS(lower(cellpath), r"^.*rbc.*$")
        OR REGEXP_CONTAINS(lower(cellpath), r"^.*red blood cell.*$"))
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep), tw=tw_in_minutes)
    rbc = gcp2df(client, query)
    return rbc


def query_ffp_trans_eicu(client, icuids_to_keep, tw_in_minutes):
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/{tw}) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/{tw}) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/{tw}) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE (REGEXP_CONTAINS(lower(cellpath), r"^.*plasma.*$")
        OR REGEXP_CONTAINS(lower(cellpath), r"^.*ffp.*$"))
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep), tw=tw_in_minutes)
    ffp = gcp2df(client, query)
    return ffp


def query_pll_trans_eicu(client, icuids_to_keep, tw_in_minutes):
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/{tw}) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/{tw}) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/{tw}) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE REGEXP_CONTAINS(lower(cellpath), r"^.*platelet.*$")
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep), tw=tw_in_minutes)
    platelets = gcp2df(client, query)
    return platelets


def query_colloid_eicu(client, icuids_to_keep, tw_in_minutes):
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/{tw}) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/{tw}) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/{tw}) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE REGEXP_CONTAINS(lower(cellpath), r"^.*colloid.*$")
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep), tw=tw_in_minutes)
    colloid = gcp2df(client, query)
    return colloid


def query_crystalloid_eicu(client, icuids_to_keep, tw_in_minutes):
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/{tw}) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/{tw}) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/{tw}) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE REGEXP_CONTAINS(lower(cellpath), r"^.*crystalloid.*$")
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep), tw=tw_in_minutes)
    crystalloid = gcp2df(client, query)
    return crystalloid


def query_comorbidity_eicu(client, icuids_to_keep):
    query = \
        """
        SELECT ad.patientunitstayid

            -- Myocardial infarction
            , MAX(CASE WHEN
                SUBSTR(ad.icd9code, 1, 3) IN ('410','412')
                THEN 1 
                ELSE 0 END) AS myocardial_infarct

            -- Congestive heart failure
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) = '428'
                OR
                SUBSTR(ad.icd9code, 1, 6) IN ('398.91','402.01','402.11','402.91','404.01','404.03',
                                '404.11','404.13','404.91','404.93')
                OR 
                SUBSTR(ad.icd9code, 1, 5) BETWEEN '425.4' AND '425.9'
                THEN 1 
                ELSE 0 END) AS congestive_heart_failure

            -- Peripheral vascular disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('440','441')
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('093.0','437.3','4471.','557.1','557.9','V43.4')
                OR
                SUBSTR(ad.icd9code, 1, 5) BETWEEN '443.1' AND '443.9'
                THEN 1 
                ELSE 0 END) AS peripheral_vascular_disease

            -- Cerebrovascular disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) BETWEEN '430' AND '438'
                OR
                SUBSTR(ad.icd9code, 1, 6) = '362.34'
                THEN 1 
                ELSE 0 END) AS cerebrovascular_disease

            -- Dementia
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) = '290'
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('294.1','331.2')
                THEN 1 
                ELSE 0 END) AS dementia

            -- Chronic pulmonary disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) BETWEEN '490' AND '505'
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('416.8','416.9','506.4','508.1','508.8')
                THEN 1 
                ELSE 0 END) AS chronic_pulmonary_disease

            -- Rheumatic disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) = '725'
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('446.5','710.0','710.1','710.2','710.3',
                                                        '710.4','714.0','714.1','714.2','714.8')
                THEN 1 
                ELSE 0 END) AS rheumatic_disease

            -- Peptic ulcer disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('531','532','533','534')
                THEN 1 
                ELSE 0 END) AS peptic_ulcer_disease

            -- Mild liver disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('570','571')
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('070.6','070.9','573.3','573.4','573.8','573.9','V42.7')
                OR
                SUBSTR(ad.icd9code, 1, 6) IN ('070.22','070.23','070.32','070.33','070.44','070.54')
                THEN 1 
                ELSE 0 END) AS mild_liver_disease

            -- Diabetes without chronic complication
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 5) IN ('250.0','250.1','250.2','250.3','250.8','250.9') 
                THEN 1 
                ELSE 0 END) AS diabetes_without_cc

            -- Diabetes with chronic complication
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 5) IN ('250.4','250.5','250.6','250.7')
                THEN 1 
                ELSE 0 END) AS diabetes_with_cc

            -- Hemiplegia or paraplegia
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('342','343')
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('334.1','344.0','344.1','344.2',
                                                        '344.3','344.4','344.5','344.6','344.9')
                THEN 1 
                ELSE 0 END) AS paraplegia

            -- Renal disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('582','585','586','V56')     
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('588.0','V42.0','V45.1')  
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('583.0', '583.1','583.2','583.3','583.4','583.5', '583.6','583.7')
                OR
                SUBSTR(ad.icd9code, 1, 6) IN ('403.01','403.11','403.91','404.02','404.03','404.12','404.13','404.92','404.93')  
                THEN 1 
                ELSE 0 END) AS renal_disease

            -- Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) BETWEEN '140' AND '172'
                OR
                SUBSTR(ad.icd9code, 1, 5) BETWEEN '174.0' AND '195.8'
                OR
                SUBSTR(ad.icd9code, 1, 3) BETWEEN '200' AND '208'
                OR
                SUBSTR(ad.icd9code, 1, 5) = '238.6'
                THEN 1 
                ELSE 0 END) AS malignant_cancer

            -- Moderate or severe liver disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 5) IN ('456.0','456.1','456.2')
                OR
                SUBSTR(ad.icd9code, 1, 5) BETWEEN '572.2' AND '572.8'
                THEN 1 
                ELSE 0 END) AS severe_liver_disease

            -- Metastatic solid tumor
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('196','197','198','199')
                THEN 1 
                ELSE 0 END) AS metastatic_solid_tumor

            -- AIDS/HIV
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('042','043','044')
                THEN 1 
                ELSE 0 END) AS aids

        FROM physionet-data.eicu_crd.diagnosis ad
        WHERE ad.patientunitstayid in ({icuids})
        GROUP BY ad.patientunitstayid
        ;
        """.format(icuids=','.join(icuids_to_keep))
    commo = gcp2df(client, query)
    return commo

