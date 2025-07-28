import pandas as pd

# Load patients table (only subject_id and dod)
patients_df = pd.read_csv(
    '/mnt/nvme4/Datasets/MIMIC-IV/physionet.org/files/mimiciv/3.0/hosp/patients.csv.gz',
    compression='gzip',
    usecols=['subject_id', 'dod']
)

# Load mimiciv_icd9 table (only subject_id, _id, text)
mimicicd9_df = pd.read_feather(
    '/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd9/mimiciv_icd9.feather',
    columns=['subject_id', '_id', 'text']
)

# Merge on subject_id and set subject_id as index
merged_df = patients_df.merge(mimicicd9_df, on='subject_id', how='inner')
merged_df = merged_df.set_index('subject_id')



