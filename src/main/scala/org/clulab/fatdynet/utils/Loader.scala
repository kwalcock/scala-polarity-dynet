package org.clulab.fatdynet.utils

import edu.cmu.dynet.{Dim, Expression, Initialize, LstmBuilder, ModelLoader, ParameterCollection}

import org.clulab.fatdynet.utils.Closer.AutoCloser

import scala.io.Source

object Loader {

  protected class ClosableModelLoader(filename: String) extends ModelLoader(filename) {
    def close(): Unit = done
  }

  // This converts an objectType and objectName into a decision about whether
  // to further process the line.  It can use the ModelLoader to do some
  // processing itself.  See falseModelFilter and loadLstm for examples.
  protected type ModelFilter = (ModelLoader, String, String) => Boolean

  protected def falseModelFilter(modelLoader: ModelLoader, objectType: String, objectName: String) = {
    // Skip these kinds of thing because they are likely a model of some kind.
    !(objectType == "#Parameter#" && objectName.matches(".*/_[0-9]+$"))
  }

  def loadExpressions(path: String, namespace: String = ""): Map[String, Expression] =
      filteredLoadExpressions(path, namespace, falseModelFilter)

  protected def filteredLoadExpressions(path: String, namespace: String = "", modelFilter: ModelFilter = falseModelFilter): Map[String, Expression] = {

    def read(line: String, modelLoader: ModelLoader, pc: ParameterCollection): Option[(String, Expression)] = {
      // See https://dynet.readthedocs.io/en/latest/python_saving_tutorial.html
      val Array(objectType, objectName, dimension, _, _) = line.split(" ")

      if (objectName.startsWith(namespace) && modelFilter(modelLoader, objectType, objectName)) {
        // Skip leading { and trailing }
        val dims = dimension.substring(1, dimension.length - 1).split(",").map(_.toInt)
        val expression = objectType match {
          case "#Parameter#" =>
            val param = pc.addParameters(Dim(dims))
            modelLoader.populateParameter(param, key = objectName)
            Expression.parameter(param)
          case "#LookupParameter#" =>
            val param = pc.addLookupParameters(dims.last, Dim(dims.dropRight(1)))
            modelLoader.populateLookupParameter(param, key = objectName)
            Expression.parameter(param)
          case _ => throw new RuntimeException(s"Unrecognized line in model file: '$line'")
        }
        Option((objectName, expression))
      }
      else
        None
    }

    val expressions = (new ClosableModelLoader(path)).autoClose { modelLoader =>
      Source.fromFile(path).autoClose { source =>
        val pc = new ParameterCollection

        source
            .getLines
            .filter(_.startsWith("#"))
            .flatMap { line => read(line, modelLoader, pc) }
            .toMap
      }
    }

    expressions
  }

  def loadLstm(path: String, namespace: String = ""): (Option[LstmBuilder], Map[String, Expression]) = {
    var lstmBuilder: Option[LstmBuilder] = None

    def lstmModelFilter(modelLoader: ModelLoader, objectType: String, objectName: String) = {
      if (objectType == "#Parameter#") {
        val matches = objectName.matches("(.*)/vanilla-lstm-builder/_[0-9]+$")

        if (matches) {
          val key = "/keith/vanilla-lstm-builder/"
          val pc = new ParameterCollection
          val layers = 1
          val inputDim = 100
          val hiddenDim = 20
          // Do I need to build a parameter collection with each call?

          // Do need to capture the name in regex
          modelLoader.populateModel(pc, key)
          lstmBuilder = Option(new LstmBuilder(1, 2, 3, pc))
          false
        }
        else
          true
      }
      else
        true
    }

    val expressions = filteredLoadExpressions(path, namespace, lstmModelFilter)

    (lstmBuilder, expressions)
  }

  def main(args: Array[String]): Unit = {
    Initialize.initialize(Map("random-seed" -> 2522620396l))

    val (builder, expressions) = loadLstm("model.dy.kwa") // , "/vanilla-lstm-builder/")

    expressions.keys.foreach(println)
  }
}
