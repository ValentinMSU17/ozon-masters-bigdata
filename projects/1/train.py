#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.metrics import log_loss

#
# Import model definition
#
from model import model, fields


#
# Logging initialization
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
  proj_id = sys.argv[1] 
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")

#
# Read dataset
#

read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)
df = df.sample(n=int(df.shape[0]/12), random_state=1)
y = df['label']
df.drop(columns=['label', 'id', 'day_number'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.33, random_state=42
)

#
# Train the model
#
model.fit(X_train, y_train)

#model_score = model.score(X_test, y_test)
model_score = log_loss(y_test, model.predict_proba(X_test), eps=1e-15)
logging.info(f"model score: {model_score:.3f}")

# save the model
dump(model, "{}.joblib".format(proj_id))