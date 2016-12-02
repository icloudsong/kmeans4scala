package com.base

import org.apache.spark.{SparkContext,SparkConf}
import org.apache.spark.mllib.clustering.{KMeans,KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object KMeansCluster {
  
  def main(args: Array[String]) {
     if (args.length < 5) {
        println("Invalid args");
        sys.exit(1);
     }
     
     val conf = new SparkConf().setAppName("Spark Mlib KMeans");
     val sc = new SparkContext(conf);
     
     //获取 原始待训练数据
     val rawTrainingData = sc.textFile(args(0));
     
     //预处理 原始待训练数据
     val parsedTrainingData = rawTrainingData.filter(!isCloumnNameLine(_)).map(line => {
       Vectors.dense(line.split(",").map(_.trim).filter(!"".equals(_)).map(_.toDouble))
     }).cache()
     
     val numClusters = args(2).toInt    //簇的个数
     val numIterations = args(3).toInt  //迭代次数
     val runTimes = args(4).toInt       //运行次数
     var clusterIndex:Int = 0           //簇序号
     
     //聚类数据 训练
     val clusters: KMeansModel = KMeans.train(parsedTrainingData, numClusters, numIterations, runTimes)
     println("Cluster Number:" + clusters.clusterCenters.length)
     println("Cluster Centers Information Overview:")
     clusters.clusterCenters.foreach(
       x => {
          println("Center Point of Cluster " + clusterIndex + ":")
          println(x)
          clusterIndex += 1
     })
     
     //获取 原始待预测数据
     val rawTestData = sc.textFile(args(1))
     
     //预处理 待预测数据
     val parsedTestData = rawTestData.map(line => {
       Vectors.dense(line.split(" ").map(_.trim).filter(!"".equals(_)).map(_.toDouble))
     })
     
     //聚类预测
     parsedTestData.collect().foreach(testDataLine => {
       val predictedClusterIndex:Int = clusters.predict(testDataLine)
       println("The data " + testDataLine.toString() + "belongs to cluster " + predictedClusterIndex)
     })
     
     println("Spark MLib K-means Clustering test finished.")
     
  }
  
  private def isCloumnNameLine(line: String):Boolean = {
    if ( line != null && line.contains("Channel") ) true
    else false
  }
  
}