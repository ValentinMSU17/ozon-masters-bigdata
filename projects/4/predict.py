#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))


#
# Read script arguments
#
try:
    model_path = sys.argv[1] 
    test_path = sys.argv[2]
    predict_path = sys.argv[3]
except:
    logging.critical("Need to pass model path, test_path, predict path")
    sys.exit(1)

logging.info(f"MODEL_PATH {model_path}")
logging.info(f"TEST_PATH {test_path}")
logging.info(f"PREDICT_PATH {predict_path}")


#
# model loading
#
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import Pipeline, PipelineModel

model = PipelineModel.load(model_path)

#
# prediction
#
dataset = spark.read.json(
    test_path, 
    multiLine=True
)

dataset_cleaned = dataset.select("id", "reviewText", "overall")
predictions = pipeline_model.transform(dataset_cleaned)
predictions.select("prediction").write.parquet(predict_path, mode="overwrite")