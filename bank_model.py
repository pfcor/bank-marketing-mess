
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

# from pyspark import SparkContext, SparkConf


# conf = SparkConf()
# conf.setAppName("modelo_bank")
# spark = SparkContext(conf=conf)


spark = SparkSession.builder.appName("BANK_MODELO").getOrCreate()

# Lendo os dados do HDFS

data = spark.read.csv(
    "hdfs://elephant:8020/user/labdata/bank.csv",
    header=True,
    sep=";",
    inferSchema=True
)


data = data.selectExpr(*["`{}` as {}".format(col, col.replace('.', '_')) for col in data.columns])


# Preparacao dos Dados

categoricalColumns = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'contact',
    'month',
    'day_of_week',
    'poutcome'
]

numericColumns = [
    'pdays',
    'previous',
    'emp_var_rate',
    'cons_price_idx',
    'cons_conf_idx',
    'euribor3m',
    'nr_employed'
]


# Criando o pipeline

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

stages = []

# transformacoes dados categoricos
for categoricalCol in categoricalColumns:
    # nomes para valores [0:n_cats-1]
    indexer = StringIndexer(
        inputCol=categoricalCol, 
        outputCol=categoricalCol+'_index'
    )
    # criando dummies
    encoder = OneHotEncoder(
        inputCol=categoricalCol+'_index',
        outputCol=categoricalCol+'_class_vec'
    )
    # inserindo estagios de transformacao
    stages += [indexer, encoder]


# indexacao da variavel resposta
indexer = StringIndexer(
    inputCol='y', 
    outputCol='label'
)
stages += [indexer]

# transformando variaveis numericas para o tipo double
for numericCol in numericColumns:
    data = data.withColumn(numericCol, data[numericCol].cast('double'))


# criando assembler, que deixa os dados no formato vetorial 
# demandado pela biblioteca ML do Spark

assembler_inputs = [categoricalCol+'_class_vec' for categoricalCol in categoricalColumns]
assembler_inputs += numericColumns
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

stages += [assembler]

# Pipeline
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(data)
pipelineModel.write().overwrite().save("data_precossing_bank_mkt")

# prepare to train
data_model = pipelineModel.transform(data)
data_model = data_model.select(["label", "features"])
# data_model.createOrReplaceTempView("data_model") 
# data_model.coalesce(1).write.mode('overwrite').csv("hdfs://elephant:8020/user/labdata/demand_models_stats.csv", header=True)

# Modelo - GBTClassifier

from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    maxDepth=2,
    maxIter=60,
    seed=420
)

gbtModel = gbt.fit(data_model)
gbtModel.write().overwrite().save("modelo_bank_mkt")

# gbt = GBTClassifier(
#     labelCol="label",
#     featuresCol="features",
#     maxDepth=2,
#     maxIter=60,
#     seed=420
# )
# pipeline = Pipeline(stages=stages)

# data_model = pipelineModel.transform(data)
# data_model = data_model.select(["label", "features"])

# model = gbt.fit(data_model)
# model.save("model_bank_marketing")

# # (trainingData, testData) = data_model.randomSplit([1.0, 0.0], seed=420)


# # Modelagem

# # Gradient Boosting Machine

# from pyspark.ml.classification import GBTClassifier

# gbt = GBTClassifier(
#     labelCol="label",
#     featuresCol="features",
#     maxDepth=2,
#     maxIter=60,
#     seed=420
# )

# gbtModel = gbt.fit(trainingData)
# gbtModel