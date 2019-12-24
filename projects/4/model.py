from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline




tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words")
stop_words = StopWordsRemover.loadDefaultStopWords("english")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words_filtered", stopWords=stop_words)
count_vectorizer = CountVectorizer(inputCol=tokenizer.getOutputCol(), outputCol="word_vector", binary=True)
lr = LinearRegression(featuresCol = count_vectorizer.getOutputCol(), labelCol='overall', regParam=0.01)
pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    count_vectorizer,
    lr
])