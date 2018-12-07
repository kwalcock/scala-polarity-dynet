package org.clulab.fatdynet.utils

import edu.cmu.dynet.{Dim, Expression, Initialize, LookupParameter, LstmBuilder, ModelLoader, ModelSaver, Parameter, ParameterCollection}

import org.clulab.fatdynet.utils.Closer.AutoCloser

import scala.io.Source

object Loader {

  protected class ClosableModelLoader(filename: String) extends ModelLoader(filename) {
    def close(): Unit = done
  }

  protected class ClosableModelSaver(filename: String) extends ModelSaver(filename) {
    def close(): Unit = done
  }

  // This converts an objectType and objectName into a decision about whether
  // to further process the line.  It can use the ModelLoader to do some
  // processing itself.  See falseModelFilter and loadLstm for examples.
  protected type ModelFilter = (ModelLoader, String, String, Array[Int]) => Boolean

  protected def falseModelFilter(modelLoader: ModelLoader, objectType: String, objectName: String, dims: Array[Int]) = {
    // Skip these kinds of thing because they are likely a model of some kind.
    !(objectType == "#Parameter#" && objectName.matches(".*/_[0-9]+$"))
  }

  def loadExpressions(path: String, namespace: String = ""): Map[String, Expression] =
      filteredLoadExpressions(path, namespace, falseModelFilter)

  protected def filteredLoadExpressions(path: String, namespace: String = "", modelFilter: ModelFilter = falseModelFilter): Map[String, Expression] = {

    def read(line: String, modelLoader: ModelLoader, pc: ParameterCollection): Option[(String, Expression)] = {
      // See https://dynet.readthedocs.io/en/latest/python_saving_tutorial.html
      val Array(objectType, objectName, dimension, _, _) = line.split(" ")
      val dims = dimension.substring(1, dimension.length - 1).split(",").map(_.toInt)

      if (objectName.startsWith(namespace) && modelFilter(modelLoader, objectType, objectName, dims)) {
        // Skip leading { and trailing }
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
    val model = new ParameterCollection
    var inputDim = -1
    var hiddenDim = -1
    var layers = 0

    def lstmModelFilter(modelLoader: ModelLoader, objectType: String, objectName: String, dims: Array[Int]) = {
      if (objectType == "#Parameter#" && objectName.matches(".*/vanilla-lstm-builder/_[0-9]+$")) {
        val param = model.addParameters(Dim(dims))
        modelLoader.populateParameter(param, key = objectName)

        // This is only going to support one model, at least one per namespace.
        if (layers == 0)
          inputDim = dims(1)
        else if (layers == 1)
          hiddenDim = dims(1)
        layers += 1
        false
      }
      else
        true
    }

    // Could throw exception rather than use option
    val expressions = filteredLoadExpressions(path, namespace, lstmModelFilter)
    val lstmBuilder =
        if (layers >= 2)
          Option(new LstmBuilder(layers - 2, inputDim, hiddenDim, model))
        else
          None

    (lstmBuilder, expressions)
  }

  def write(filename: String): Unit = {
    val VOC_SIZE = 3671
    val W2V_SIZE = 1234

    val WEM_DIMENSIONS = 100
    val NUM_LAYERS = 1
    val FF_HIDDEN_DIM = 10
    val HIDDEN_DIM = 20

    val pc = new ParameterCollection
    val model = new ParameterCollection

    val W_p: Parameter = pc.addParameters(Dim(Seq(FF_HIDDEN_DIM, HIDDEN_DIM)))
    val b_p: Parameter = pc.addParameters(Dim(Seq(FF_HIDDEN_DIM)))
    val V_p: Parameter = pc.addParameters(Dim(Seq(1, FF_HIDDEN_DIM)))

    val w2v_wemb_lp: LookupParameter = pc.addLookupParameters(W2V_SIZE, Dim(Seq(WEM_DIMENSIONS)))
    val missing_wemb_lp: LookupParameter = pc.addLookupParameters(VOC_SIZE, Dim(Seq(WEM_DIMENSIONS)))

    val builder = new LstmBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, model)

    (new ClosableModelSaver(filename)).autoClose { saver =>
      saver.addParameter(W_p, "/W")
      saver.addParameter(b_p, "/b")
      saver.addParameter(V_p, "/V")
      saver.addLookupParameter(w2v_wemb_lp, "/w2v-wemb")
      saver.addLookupParameter(missing_wemb_lp, "/missing-wemb")
      saver.addModel(model)
    }
  }

  def read(filename: String): Unit = {
    val (optionBuilder, expressions) = loadLstm(filename)

    expressions.keys.foreach(println)

    val builder = optionBuilder.get
    val V = expressions("/V")
    val W = expressions("/W")
    val b = expressions("/b")

    val pc = new ParameterCollection
    // Can this be different?
    val WEM_DIMENSIONS = 100
    val w2v_wemb_lp: LookupParameter = pc.addLookupParameters(1000 /*584550 1579375*/, Dim(Seq(WEM_DIMENSIONS)))

    val inputs = 0.until(100) map { Expression.lookup(w2v_wemb_lp, _)}

    builder.newGraph()
    builder.startNewSequence()

    val states = inputs map {
      w => builder.addInput(w)
    }
    val selected = states.last
    val prediction = Expression.logistic(V * (W * selected + b))

    println(prediction)
  }

  def main(args: Array[String]): Unit = {
    val filename = "model.dy.kwa"

    Initialize.initialize(Map("random-seed" -> 2522620396l))
    write(filename)
    read(filename)
  }
}
