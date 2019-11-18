#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd
import numpy as np

sys.path.append('.')
from model import fields

logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

model = load("2.joblib")

fields.remove('label')
drop_columns = ['cf1', 'cf10', 'cf20', 'cf21', 'cf22'] + ['id', 'day_number']


for line in sys.stdin:
    df_row = line.strip().split('\t')
    data_dict = dict(zip(fields, df_row))
    for key in data_dict.keys():
        if (data_dict[key] == '\\N') | (data_dict[key] == ''):
            data_dict[key] = np.nan
            continue
        if key.startswith('i'):
            data_dict[key] = int(data_dict[key])

    df = pd.DataFrame(data_dict, index=[0])
    ids = df['id']
    df.drop(columns=drop_columns, inplace=True)
    
    if (df['if1'][0] > 20) & (df['if1'][0] < 40):
        pred = model.predict_proba(df)[:,1]
        print(int(ids), pred[0])