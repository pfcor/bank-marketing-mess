df = spark.read.options(header='true').csv("train_titanic.csv")

from pyspark.sql.functions import avg, col
for i in df.columns:
  print(i+' '+str(df.filter(col(i).isNull()).count()))

avg_age = df.agg(avg(col("Age"))).collect()[0][0]
print(avg_age)

df = df.withColumn("Age", df["Age"].cast("double"))
df_imp = df.fillna(avg_age, subset="Age")

for i in df_imp.columns:
  print(i+' '+str(df_imp.filter(col(i).isNull()).count()))


from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

## Defining stages for categorical feature (StringIndexer + Dummies creating)
categoricalColumns = ["Sex"]

stages = [] 
for categoricalCol in categoricalColumns:
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
  stages += [stringIndexer, encoder]


label_stringIdx = StringIndexer(inputCol = "Survived", outputCol = "label")
stages += [label_stringIdx]


numericCols = ["Age", "SibSp", "Parch", "Fare"]
for i in numericCols:
  df_imp = df_imp.withColumn(i, df_imp[i].cast("double"))


assemblerInputs = ['SexclassVec'] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df_imp)

df_imp = pipelineModel.transform(df_imp)

## Keep relevant columns
selectedcols = ["label", "features"]
df_imp = df_imp.select(selectedcols)

(trainingData, testData) = df_imp.randomSplit([0.7, 0.3], seed = 42)
print(trainingData.count())
print(testData.count())


## Import logistic regression
from pyspark.ml.classification import LogisticRegression
## Defining initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

## Train model with Training Data
lrModel = lr.fit(trainingData)

## Predicting in test data
predictions = lrModel.transform(testData)

## Evaluate using AUC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features",)


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
pipeline_rf = Pipeline(stages=[rf])
paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, 
	[2,3,4,5,6,7]).addGrid(rf.numTrees, [100,300]).build()


evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction")
crossval = CrossValidator(estimator=pipeline_rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

## Fitting the CV
CV_model = crossval.fit(trainingData)

## Printing best model
print(CV_model.bestModel.stages[0])


test_pred = CV_model.transform(testData)
print(evaluator.getMetricName(), evaluator.evaluate(test_pred))
