package org.clulab.fatdynet.utils

import edu.cmu.dynet.{Dim, Expression, Initialize, ModelLoader, ParameterCollection}

import org.clulab.fatdynet.utils.Closer.AutoCloser

import scala.io.Source

object Loader {

  class ClosableModelLoader(filename: String) extends ModelLoader(filename) {

    def close(): Unit = done
  }

  // loadModel?
  def load(path: String, nameSpace: String = "/"): Map[String, Expression] = {

    def read(line: String, modelLoader: ModelLoader, pc: ParameterCollection): (String, Expression) = {
      val Array(typ, name, dimensions, _, _) = line.split(" ")
      // Skip leading { and trailing }
      val dims = dimensions.substring(1, dimensions.length - 1).split(",").map(_.toInt)
      val expression = typ match {
        case "#Parameter#" =>
          val param = pc.addParameters(Dim(dims))
          modelLoader.populateParameter(param, key = name)
          Expression.parameter(param)
        case "#LookupParameter#" =>
          val param = pc.addLookupParameters(dims(1), Dim(dims(0)))
          modelLoader.populateLookupParameter(param, key = name)
          Expression.parameter(param)
        case _ => throw new RuntimeException("Unrecognized line in model file")
      }

      (name, expression)
    }

    val expressions = (new ClosableModelLoader(path)).autoClose { loader =>
      Source.fromFile(path).autoClose { source =>
        val pc = new ParameterCollection

        source
            .getLines
            .filter(_.startsWith("#"))
            .filter(_.contains(nameSpace))
            .map { line => read(line, loader, pc) }
            .toMap
      }
    }

    expressions
  }

  def main(args: Array[String]): Unit = {
    Initialize.initialize(Map("random-seed" -> 2522620396l))

    val expressions: Map[String, Expression] = load("model.dy") // , "/vanilla-lstm-builder/")

    expressions.keys.foreach(println)
  }
}
