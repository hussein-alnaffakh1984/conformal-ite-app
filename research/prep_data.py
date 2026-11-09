"""Build the 18 real covariates from Diabetes 130-US Hospitals (Strack et al. 2014).
Rows are kept in encounter_id order (used as the stream order in the online study)."""
import sys, numpy as np, pandas as pd
src = sys.argv[1]; out = sys.argv[2]
df = pd.read_csv(src).replace('?', np.nan)
n_raw = len(df)
age_mid = {f'[{10*k}-{10*k+10})': 10*k+5 for k in range(10)}
df['age_mid'] = df['age'].map(age_mid)
df = df[df['gender'].isin(['Male', 'Female'])]          # drops 3 'Unknown/Invalid'
df = df.sort_values('encounter_id').reset_index(drop=True)
num = ['age_mid','time_in_hospital','num_lab_procedures','num_procedures','num_medications',
       'number_outpatient','number_emergency','number_inpatient','number_diagnoses',
       'admission_type_id','discharge_disposition_id','admission_source_id']
X = df[num].astype(float).copy()
X['male'] = (df['gender'] == 'Male').astype(float)
for r in ['Caucasian','AfricanAmerican','Hispanic','Asian','Other']:
    X['race_'+r] = (df['race'] == r).astype(float)
names = list(X.columns)
Xs = ((X - X.mean()) / X.std()).to_numpy()
np.savez_compressed(out, X=Xs, names=np.array(names), encounter_id=df['encounter_id'].to_numpy(),
                    A_real=(df['change']=='Ch').astype(int).to_numpy(),
                    Y_real=(df['readmitted']=='<30').astype(int).to_numpy())
print('raw rows', n_raw, '-> kept', len(df), '| covariates', len(names))
print(names)
print('real A rate %.3f  real Y rate %.3f' % ((df['change']=='Ch').mean(), (df['readmitted']=='<30').mean()))
