#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd
import numpy as np

sys.path.append('.')
from model import fields

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("1.joblib")


#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

fields.remove('label')

#read and infere
read_opts=dict(
        sep='\t', names=fields, index_col=False, header=None,
        iterator=True, chunksize=1000
)
drop_columns = ['cf1', 'cf10', 'cf20', 'cf21', 'cf22'] + ['id', 'day_number']

for df in pd.read_csv(sys.stdin, **read_opts):
    ids = df['id']
    df.drop(columns=drop_columns, inplace=True)
    pred = model.predict_proba(df)[:,1]
    out = zip(ids, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))