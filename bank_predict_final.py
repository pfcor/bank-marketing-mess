# # BANK MARKETING
# 
# 
# Membros:
# - Anderson
# - Caio Viera
# - Pedro Correia
# 
# 

# misc
import datetime as dt
timestamp = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d')

# init
from pyspark.context import SparkContext, SparkConf
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("BANK_MODELO").getOrCreate()

# carregando modelo
from pyspark.ml import PipelineModel
pipelineModel = PipelineModel.load('hdfs://elephant:8020/user/labdata/model/bank-pipeline-model-res/')

# carregando dados
data = spark.read.csv(
    "hdfs://elephant:8020/user/labdata/new-data.csv",
    header=True,
    sep=";",
    inferSchema=True
)
data = data.selectExpr(*["`{}` as {}".format(col, col.replace('.', '_')) for col in data.columns])

# fazendo as predições
predictions = pipelineModel.transform(data)

# salvando predições
predictions.select('label', 'prediction', predictions['features'].cast('string')).write.csv('hdfs://elephant:8020/user/labdata/predictions/{}/predictions'.format(timestamp))
# salvando metricas
predictions.select('label', 'prediction').createOrReplaceTempView('predictions')
spark.sql("""
SELECT
    round((tp+tn)/(tp+tn+fp+fn), 4) as accuracy,
    round(tp/(tp+fp), 4) as precision,
    round(tp/(tp+fn), 4) as recall
FROM (
    SELECT
        sum(tn) as tn,
        sum(tp) as tp,
        sum(fn) as fn,
        sum(fp) as fp
    FROM (
        SELECT
            case when label = 0 and prediction = 0 then 1 else 0 end as tn,
            case when label = 1 and prediction = 1 then 1 else 0 end as tp,
            case when label = 1 and prediction = 0 then 1 else 0 end as fn,
            case when label = 0 and prediction = 1 then 1 else 0 end as fp
        FROM
            predictions
    )
)
""").write.csv('hdfs://elephant:8020/user/labdata/predictions/{}/metrics'.format(timestamp))
#""").write.csv('/home/pfcor/bank-marketing/predictions/{}/metrics'.format(timestamp))
