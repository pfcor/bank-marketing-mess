# # BANK MARKETING
# 
# 
# Membros:
# - Anderson
# - Caio Viera
# - Pedro Correia
# 
# 



from pyspark.context import SparkContext, SparkConf
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BANK_MODELO").getOrCreate()

# Lendo os dados do HDFS
data = spark.read.csv(
    "hdfs://elephant:8020/user/labdata/bank_small.csv",
    header=True,
    sep=";",
    inferSchema=True
)
data = data.selectExpr(*["`{}` as {}".format(col, col.replace('.', '_')) for col in data.columns])

# with open('prediction_log.txt', 'w') as logFile:
# 	logFile.write('testeeeee')

# process
from pyspark.ml import Pipeline, PipelineModel
pipelineModel = PipelineModel.read().load('data_precossing_bank_mkt')

data_model = pipelineModel.transform(data)
data_model = data_model.select(["label", "features"])

# predict
from pyspark.ml.classification import GBTClassifier
gbtModel = GBTClassifier.load("modelo_bank_mkt")

predictions_gbt = gbtModel.transform(data_model)

#evaluate
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction",
    metricName="accuracy"
)

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label", 
    rawPredictionCol="rawPrediction"
)

accuracy_gbt = evaluator_accuracy.evaluate(predictions_gbt)
print(f'Accuracy:         {accuracy_gbt:.4f}')
auc_gbt = evaluator_auc.evaluate(predictions_gbt)
print(f'areaUnderROC:     {auc_gbt:.4f}')

predictions_gbt.write.mode('overwrite').csv("hdfs://elephant:8020/user/labdata/predictions.csv", header=True)

# with open('hdfs://elephant:8020/user/labdata/prediction_log.txt', 'w') as logFile:
# 	logFile.write('testeeeee')