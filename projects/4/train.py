#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

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
    train_path = sys.argv[1]
    model_path = sys.argv[2] 
except:
    logging.critical("Need to pass both train dataset path and model path")
    sys.exit(1)


logging.info(f"TRAIN_PATH {train_path}")
logging.info(f"MODEL_PATH {model_path}")


#
# model importing
#
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline


#
# training
#
dataset = spark.read.json(
    train_path, 
    multiLine=True
)
dataset_cleaned = dataset.select("id", "reviewText", "overall")
pipeline_model = pipeline.fit(dataset_cleaned)

pipeline_model.write().overwrite().save(model_path)