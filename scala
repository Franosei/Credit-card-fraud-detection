import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._
import org.apache.spark.mllib.evaluation._
/////////////////////////////////////////////////////////

val rawData = sc.textFile("/home/ubuntulab/Downloads/data/creditcard.csv")
val header = rawData.first() 
val data_rdd = rawData.filter(row => row != header)

val data = data_rdd.map { line =>
  val values = line.split(',').map(_.toDouble)
  val featureVector = Vectors.dense(values.init)
  val label = values.last
  LabeledPoint(label, featureVector)
}

val Array(trainData, valData, testData) =
  data.randomSplit(Array(0.8, 0.1, 0.1))

trainData.cache() // subset of dataset used for training
valData.cache() // subset of dataset used for optimization of hyperparameters
testData.cache() // subset of dataset used for final evaluation ("testing")


//////////////fitting the decision tree model///////////////////
val model = DecisionTree.trainClassifier(
  trainData, 7, Map[Int,Int](), "gini", 4, 100)

println(model.toDebugString)

def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]):
  MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

////////////////////Confusion matrix for unbalanced data/////

val predictionAndLabels = testData.map{ case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

val metrics = new MulticlassMetrics(predictionAndLabels)
println("Confusion matrix:")
println(metrics.confusionMatrix)
  
/////////evaluating precison and recall of the validation dataset///
val metrics = getMetrics(model, valData)
(0 until 1).map(
  label => (metrics.precision(label), metrics.recall(label))
).foreach(println)


(0 until 1).map(
  label => (metrics.precision(label), metrics.recall(label))
).foreach(println)

///////////////////////////////////////////////////////////////////
def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
  val countsByCategory = data.map(_.label).countByValue()
  val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
  counts.map(_.toDouble / counts.sum)
}

////// compare the model to a naive prediction based on prior class probabilites ---

val trainPriorProbabilities = classProbabilities(trainData)
val valPriorProbabilities = classProbabilities(valData)
trainPriorProbabilities.zip(valPriorProbabilities).map {
  case (trainProb, valProb) => trainProb * valProb
}.sum

/////////optimize the hyperparameters of the model/////////////// 

val evaluations =
  for (impurity <- Array("gini", "entropy");
    depth <- Array(10, 20, 30);
    bins <- Array(50, 100, 300))
  yield {
    val model = DecisionTree.trainClassifier(
      trainData, 7, Map[Int,Int](), impurity, depth, bins)
    val predictionsAndLabels = valData.map(example =>
      (model.predict(example.features), example.label)
    )  
    val accuracy =
      new MulticlassMetrics(predictionsAndLabels).accuracy
    ((impurity, depth, bins), accuracy) }

evaluations.sortBy(_._2).reverse.foreach(println)

//////////////////////////////////////////////////////////////////

/////////////////Balancing the the training set///////////////////

val df_balance = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("/home/ubuntulab/Downloads/data/creditcard.csv")

def underSampleDf(df:DataFrame) = {
    val fraudDf = df.filter("Class=1.0")
    val nonFraudDf = df.filter("Class=0.0")
    //random sample the nonFraud to match the value of fraud
    val sampleRatio = fraudDf.count().toDouble / df.count().toDouble
    val nonFraudSampleDf = nonFraudDf.sample(false, sampleRatio)
    fraudDf.unionAll(nonFraudSampleDf)
  }
val df_sample = underSampleDf(df_balance)
val df1Sample = df_sample.rdd.map(_.mkString(","))

val data1 = df1Sample.map { line =>
  val values = line.split(',').map(_.toDouble)
  val featureVector = Vectors.dense(values.init)
  val label = values.last
  LabeledPoint(label, featureVector)
}

val Array(trainData1) =
  data1.randomSplit(Array(0.1))
trainData1.cache()

//////////////////Decision tree for balance data//////////////
val model1 = DecisionTree.trainClassifier(
  trainData1, 7, Map[Int,Int](), "gini", 4, 100)

println(model1.toDebugString)

def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]):
  MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

val metrics = getMetrics(model1, valData)

(0 until 1).map(
  label => (metrics.precision(label), metrics.recall(label))
).foreach(println)


////////////////////////Confusion Matrix balnaced data//////////
val predictionAndLabels = testData.map{ case LabeledPoint(label, features) =>
  val prediction = model1.predict(features)
  (prediction, label)
}

val metrics = new MulticlassMetrics(predictionAndLabels)
println("Confusion matrix:")
println(metrics.confusionMatrix)

/////////////////////////////////////////////////////////////

def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
  val countsByCategory = data.map(_.label).countByValue()
  val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
  counts.map(_.toDouble / counts.sum)
}

val trainPriorProbabilities = classProbabilities(trainData1)
val valPriorProbabilities = classProbabilities(valData)
trainPriorProbabilities.zip(valPriorProbabilities).map {
  case (trainProb, valProb) => trainProb * valProb
}.sum

/////////optimize the hyperparameters of the model/////////////// 

val evaluations =
  for (impurity <- Array("gini", "entropy");
    depth <- Array(10, 20, 30);
    bins <- Array(50, 100, 300))
  yield {
    val model = DecisionTree.trainClassifier(
      trainData1, 7, Map[Int,Int](), impurity, depth, bins)
    val predictionsAndLabels = valData.map(example =>
      (model.predict(example.features), example.label)
    )  
    val accuracy =
      new MulticlassMetrics(predictionsAndLabels).accuracy
    ((impurity, depth, bins), accuracy) }

evaluations.sortBy(_._2).reverse.foreach(println)
