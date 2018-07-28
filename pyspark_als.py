df = spark.read.options(header='false',sep='\t').csv("base_recommendation.txt")

oldColumns = df.schema.names
newColumns = ["userId", "itemId", "rating"]

df = df.withColumnRenamed(oldColumns[0], newColumns[0]).withColumnRenamed(oldColumns[1], newColumns[1]).withColumnRenamed(oldColumns[2], newColumns[2])
df.printSchema()

df = df.withColumn("userId", df["userId"].cast("string"))
df = df.withColumn("itemId", df["itemId"].cast("string"))
df.printSchema()

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from pyspark.ml.feature import StringIndexer

indexer_acc = StringIndexer(inputCol="userId", outputCol="userIndex")
indexer_acc_fitted = indexer_acc.fit(df)
df = indexer_acc_fitted.transform(df)

indexer_mer = StringIndexer(inputCol="itemId", outputCol="itemIndex")
indexer_mer_fitted = indexer_mer.fit(df)
df = indexer_mer_fitted.transform(df)

df = df.withColumn("rating", df["rating"].cast("double"))
als = ALS(regParam=0.01, userCol="userIndex", itemCol="itemIndex",  ratingCol="rating")

(training, test) = df.randomSplit([1.0, 0.0])

model = als.fit(training)

predictions = model.transform(training)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


import numpy
lista_itens = list(numpy.unique(df.toPandas().itemId))

def to_pred(acc):
  df_pred = sc.parallelize(map(lambda x: (x, acc), lista_itens)).toDF(["itemId", "userId"])
  df_pred = indexer_acc_fitted.transform(df_pred)
  df_pred = indexer_mer_fitted.transform(df_pred)
  df_pred = model.transform(df_pred)
  return df_pred.orderBy('prediction', ascending=False)


pred = to_pred('146')
pred.take(10)


from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
als = ALS(userCol="userIndex", itemCol="itemIndex",  ratingCol="rating")
pipeline = Pipeline(stages=[als])

paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.01, 0.1]).addGrid(als.rank, [5, 10]).build()

evaluator = RegressionEvaluator(labelCol='rating',predictionCol='prediction', metricName="rmse")    

## Creating 3 fold cross validation using the pipeline and Grid Search

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

## Fitting the model
CV_model = crossval.fit(training)

predictions = CV_model.transform(training)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

