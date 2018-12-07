package org.clulab.fatdynet.utils

import edu.cmu.dynet.{Dim, Expression, Initialize, LstmBuilder, ModelLoader, ParameterCollection}

import org.clulab.fatdynet.utils.Closer.AutoCloser

import scala.io.Source

object Loader {

  class ClosableModelLoader(filename: String) extends ModelLoader(filename) {
    def close(): Unit = done
  }

  // Add version with it's own, open model loader than can be used later
  def loadExpressions(path: String, nameSpace: String = "/"): Map[String, Expression] = {

    def read(line: String, modelLoader: ModelLoader, pc: ParameterCollection): (String, Expression) = {
      // See https://dynet.readthedocs.io/en/latest/python_saving_tutorial.html
      val Array(objectType, objectName, dimension, _, _) = line.split(" ")
      // Skip leading { and trailing }
      val dims = dimension.substring(1, dimension.length - 1).split(",").map(_.toInt)
      val expression = objectType match {
        case "#Parameter#" =>
          val param = pc.addParameters(Dim(dims))
          param.dim().get(0)
          modelLoader.populateParameter(param, key = objectName)
          Expression.parameter(param)
        case "#LookupParameter#" => // if name begins with modelName?
          val param = pc.addLookupParameters(dims(1), Dim(dims(0)))
          modelLoader.populateLookupParameter(param, key = objectName)
          Expression.parameter(param)
        case _ => throw new RuntimeException("Unrecognized line in model file")
      }

      (objectName, expression)
    }

    val expressions = (new ClosableModelLoader(path)).autoClose { modelLoader =>
      Source.fromFile(path).autoClose { source =>
        val pc = new ParameterCollection

        source
            .getLines
            .filter(_.startsWith("#"))
            .filter(_.contains(nameSpace)) // should check name for this, not entire line
            .map { line => read(line, modelLoader, pc) }
            .toMap
      }
    }

    expressions
  }

//  def loadLstm(expressions: Expression, ) = ???
  // from a list of expressions?
  // returns lstmBuilder and expressions?


  def main(args: Array[String]): Unit = {
    Initialize.initialize(Map("random-seed" -> 2522620396l))

    val expressions: Map[String, Expression] = loadExpressions("model.dy") // , "/vanilla-lstm-builder/")

    val numLayers = expressions("/V").dim.get(0) // find lstm and _0, _1, _2, how many are there, and subtract input and output
    val wemDimensions = expressions("/wemb").dim.get(0)
    val hiddenDim = expressions("/W").dim.get(1)
    val pc = new ParameterCollection()


    // It should return lstm builder
    val builder = new LstmBuilder(numLayers, wemDimensions, hiddenDim, pc)

    expressions.keys.foreach(println)
  }
}
