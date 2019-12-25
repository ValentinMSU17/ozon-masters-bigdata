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
    v_from = sys.argv[1] 
    v_to = sys.argv[2]
    df_path = sys.argv[3]
    result_path = sys.argv[4]
except:
    logging.critical("Need to pass v_from, v_to, df_path, result_path")
    sys.exit(1)

logging.info(f"V_FROM {v_from}")
logging.info(f"V_TO {v_to}")
logging.info(f"DF_PATH {df_path}")
logging.info(f"RESULT_PATH {result_path}")

#
# model loading
#
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.sql.types import *
import pyspark.sql.functions as f


#
# custom programm
#
schema = StructType(fields=[
    StructField("user_id", IntegerType()),
    StructField("follower_id", IntegerType())
])

df = spark.read\
          .schema(schema)\
          .format("csv")\
          .option("sep", "\t")\
          .load(df_path).cache()

def shortest_path(v_from, v_to, df, max_path_length=10):
    """
        v_from - исходная вершина
        v_to - целевая вершина
        df - Spark DataFrame с ребрами графа
        max_path_length - максимальная длина пути
        
        Возвращает: pyspark.sql.DataFrame, состоящий из одного столбца с найдеными путями
    """
    temp_df = df.filter(df.follower_id == v_from)
    temp_df = temp_df.select(f.col('user_id').alias('last_neighbour'), 
                             f.col('follower_id').alias('path'))

    for i in range(max_path_length):
        if temp_df.filter(temp_df.last_neighbour.isin(v_to)).count() > 0:
            result_df = temp_df.filter(temp_df.last_neighbour.isin(v_to))\
                               .select(f.concat('path', f.lit(','), 'last_neighbour').alias('path'))
            return result_df
        temp_df = temp_df.join(df, temp_df.last_neighbour==df.follower_id, how="inner",)\
                         .select(f.column('user_id').alias('last_neighbour'),
                                 f.concat('path', f.lit(','), 'last_neighbour').alias('path'))
        
result_df = shortest_path(v_from, v_to, df)
result_df.select("path").write.mode("overwrite").text(result_path)
