## one hot encoding and Sparse vectors

# sample OHE creation and sparse representation

def one_hot_encoding(raw_feats, ohe_dict_broadcast, num_ohe_feats):
    #return SparseVector(num_ohe_feats, [(ohe_dict_broadcast[(featID, value)],1) for (featID, value) in raw_feats])
    retObj = SparseVector(num_ohe_feats, [(ohe_dict_broadcast.value[ftrTpl], 1.) for ftrTpl in raw_feats])
    return retObj
# Calculate the number of features in sample_ohe_dict_manual
num_sample_ohe_feats = len(sample_ohe_dict_manual.keys())
sample_ohe_dict_manual_broadcast = sc.broadcast(sample_ohe_dict_manual)

# Run one_hot_encoding() on sample_one.  Make sure to pass in the Broadcast variable.
sample_one_ohe_feat = one_hot_encoding(sample_one, sample_ohe_dict_manual_broadcast, num_sample_ohe_feats)

print sample_one_ohe_feat

### APPLY OHE TO DATASET
def ohe_udf_generator(ohe_dict_broadcast):
    length = len(ohe_dict_broadcast.value.keys())
    return udf(lambda x: one_hot_encoding(x, ohe_dict_broadcast, length), VectorUDT())

sample_ohe_dict_udf = ohe_udf_generator(sample_ohe_dict_manual_broadcast)
sample_ohe_df = sample_data_df.select(sample_ohe_dict_udf('features'))
sample_ohe_df.show(truncate=False)


### Construct OHE dictionary

#data frame with rows of (feature ID, category)
from pyspark.sql.functions import explode
sample_distinct_feats_df = (sample_data_df
                              .select(explode('features'))
                              .distinct())

#OHE dictionary from distinct features
sample_ohe_dict = (sample_distinct_feats_df
                     .rdd
                     .map(lambda r: tuple(r[0]))
                     .zipWithIndex()
                     .collectAsMap())


#automated dictionary creation
def create_one_hot_dict(input_df):
    retObj = (input_df
               .select(explode('features'))
               .distinct()
               .rdd
               .map(lambda r: tuple(r[0]))
               .zipWithIndex()
               .collectAsMap())
    return retObj

sample_ohe_dict_auto = create_one_hot_dict(sample_data_df)
print sample_ohe_dict_auto


######### PARSE CTR AND GENERATE OHE FEATURES
#download data from here
criteo_url = 'http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz'
raw_df = sqlContext.read.text(downloaded_data_file).withColumnRenamed("value", "text")

#loading and splitting the data
weights = [.8, .1, .1]
seed = 42

# Use randomSplit with weights and seed
raw_train_df, raw_validation_df, raw_test_df = raw_df.randomSplit(weights, seed)

# Cache and count the DataFrames
n_train = raw_train_df.cache().count()
n_val   = raw_validation_df.cache().count()
n_test  = raw_test_df.cache().count()
print n_train, n_val, n_test, n_train + n_val + n_test
raw_df.show(1)


# extract features
def parse_point(point):
    parsedStr = point.split(",")
#     print "parse_point: parsedStr: %s" % (parsedStr)    
    
    retObj = [(ix - 1, parsedStr[ix]) for ix in xrange(1, len(parsedStr))]
    return retObj

print raw_df.select('text').first()[0]
print parse_point(raw_df.select('text').first()[0])

## extracting features continued
from pyspark.sql.functions import udf, split
from pyspark.sql.types import ArrayType, StructType, StructField, LongType, StringType

parse_point_udf = udf(parse_point, ArrayType(StructType([StructField('_1', LongType()),
                                                         StructField('_2', StringType())])))

def parse_raw_df(raw_df):
    newDf = raw_df.select(split('text', ",")[0].cast("double").alias('label'),
                          parse_point_udf('text').alias("feature"))

    retObj = newDf.cache()
    return retObj

# Parse the raw training DataFrame
parsed_train_df =parse_raw_df(raw_train_df.select('text'))
from pyspark.sql.functions import (explode, col)
num_categories = (parsed_train_df
                    .select(explode('feature').alias('feature'))
                    .distinct()
                    .select(col('feature').getField('_1').alias('featureNumber'))
                    .groupBy('featureNumber')
                    .sum()
                    .orderBy('featureNumber')
                    .collect())

 
print num_categories[2][1]


# Create OHE dictionary from dataset
ctr_ohe_dict = create_one_hot_dict(parsed_train_df.select(parsed_train_df.feature.alias('features')))
num_ctr_ohe_feats = len(ctr_ohe_dict.keys())
print num_ctr_ohe_feats
print ctr_ohe_dict[(0, '')]

#Apply OHE to dataset
ohe_dict_broadcast = sc.broadcast(ctr_ohe_dict)
ohe_dict_udf = ohe_udf_generator(ohe_dict_broadcast)
ohe_train_df = (parsed_train_df
                  .select('label', ohe_dict_udf(parsed_train_df.feature.alias('features')).alias('features'))
                  .cache())

print ohe_train_df.count()
print ohe_train_df.take(1)


## HAndling Unseen features
def one_hot_encoding(raw_feats, ohe_dict_broadcast, num_ohe_feats):
    retObj = SparseVector(num_ohe_feats, [(ohe_dict_broadcast.value[ftrTpl], 1.) for ftrTpl in raw_feats \
                                           if ohe_dict_broadcast.value.has_key(ftrTpl)])
    return retObj

ohe_dict_missing_udf = ohe_udf_generator(ohe_dict_broadcast)
parsed_validation_df = parse_raw_df(raw_validation_df.select('text'))
ohe_validation_df = (parsed_validation_df
                  .select('label', ohe_dict_missing_udf(parsed_validation_df.feature.alias('features')).alias('features'))
                  .cache())

ohe_validation_df.count()
ohe_validation_df.show(1, truncate=False)

## CTR PREDICTION AND LOG LOSS EVALUATION

# LOGISTIC REGRESSION
standardization = False
elastic_net_param = 0.0
reg_param = .01
max_iter = 20

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter = max_iter, regParam = reg_param, elasticNetParam = elastic_net_param, 
                        standardization = standardization)

lr_model_basic = lr.fit(ohe_train_df)

print 'intercept: {0}'.format(lr_model_basic.intercept)
print 'length of coefficients: {0}'.format(len(lr_model_basic.coefficients))
sorted_coefficients = sorted(lr_model_basic.coefficients)[:5]

# log loss
from pyspark.sql.functions import when, log, col
epsilon = 1e-16

def add_log_loss(df):
    newDF = df.select("*", when(df.label == 1, 0. - log(df.p + epsilon)).\
                           otherwise(0. - log(1. - df.p + epsilon)).alias('log_loss'))
    retObj = newDF
    return retObj  
add_log_loss(example_log_loss_df).show()

# Baseline Log-Loss
# Note that our dataset has a very high click-through rate by design
# In practice click-through rate can be one to two orders of magnitude lower

from pyspark.sql.functions import lit
class_one_frac_train = (ohe_train_df
                          .groupBy()
                          .mean('label')
                          .collect())[0][0]
# print class_one_frac_train
# print type(class_one_frac_train)
print 'Training class one fraction = {0:.3f}'.format(class_one_frac_train)

log_loss_tr_base = (add_log_loss(ohe_train_df.select('*', lit(class_one_frac_train).alias('p')))
                          .groupBy()
                          .mean('log_loss')
                          .collect())[0][0]
print 'Baseline Train Logloss = {0:.3f}\n'.format(log_loss_tr_base)


## predicted probability
from pyspark.sql.types import DoubleType
from math import exp #  exp(-t) = e^-t

def add_probability(df, model):
    """Adds a probability column ('p') to a DataFrame given a model"""
    coefficients_broadcast = sc.broadcast(model.coefficients)
    intercept = model.intercept

    def get_p(features):
        """Calculate the probability for an observation given a list of features.

        Note:
            We'll bound our raw prediction between 20 and -20 for numerical purposes.

        Args:
            features: the features

        Returns:
            float: A probability between 0 and 1.
        """
        # Compute the raw value
        raw_prediction = intercept + coefficients_broadcast.value.dot(features)
        # Bound the raw value between 20 and -20
        raw_prediction = min(20, max(-20, raw_prediction))
        # Return the probability
        probability = 1.0 / (1 + exp(- raw_prediction))
        return probability
    get_p_udf = udf(get_p, DoubleType())
    return df.withColumn('p', get_p_udf('features'))   
  
    

add_probability_model_basic = lambda df: add_probability(df, lr_model_basic)
training_predictions = add_probability_model_basic(ohe_train_df).cache()

training_predictions.show(5)


## evaluate the model
def evaluate_results(df, model, baseline=None):
    if (model != None):    
        with_probability_df = add_probability(df, model)
    
    else:
    
        with_probability_df = df.withColumn('p', lit(baseline))
    with_log_loss_df = add_log_loss(with_probability_df)
    log_loss = with_log_loss_df.groupBy().mean('log_loss').collect()[0][0]
    return log_loss

    

log_loss_train_model_basic = evaluate_results(ohe_train_df, lr_model_basic)
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(log_loss_tr_base, log_loss_train_model_basic))



#Validation Log-loss
log_loss_val_base = evaluate_results(ohe_validation_df, None, class_one_frac_train)

log_loss_val_l_r0 = evaluate_results(ohe_validation_df, lr_model_basic)
print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(log_loss_val_base, log_loss_val_l_r0))










