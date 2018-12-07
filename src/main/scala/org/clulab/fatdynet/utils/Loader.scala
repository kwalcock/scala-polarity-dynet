package org.clulab.fatdynet.utils

import edu.cmu.dynet.{
  Initialize,

  Dim,
  Expression,

  LookupParameter,
  Parameter,
  ParameterCollection,

  // These other, unused builders to be addedd

  FastLstmBuilder,
  LstmBuilder,
    CompactVanillaLSTMBuilder,
    CoupledLstmBuilder,
    VanillaLstmBuilder,

  TreeLSTMBuilder, // abstract
    UnidirectionalTreeLSTMBuilder,
    BidirectionalTreeLSTMBuilder,

  RnnBuilder, // abstract
    SimpleRnnBuilder,

  GruBuilder,

  ModelLoader,
  ModelSaver,
}

import org.clulab.fatdynet.utils.Closer.AutoCloser

import scala.io.Source

/**
  * See https://dynet.readthedocs.io/en/latest/python_saving_tutorial.html
  */
object Loader {

  protected class ClosableModelLoader(filename: String) extends ModelLoader(filename) {
    def close(): Unit = done
  }

  protected class ClosableModelSaver(filename: String) extends ModelSaver(filename) {
    def close(): Unit = done
  }

  /** This converts an objectType and objectName into a decision about whether
    * to further process the line.  It can use the ModelLoader to do some
    * processing itself.  See falseModelFilter and loadLstm for examples.
    */
  protected type ModelFilter = (ModelLoader, String, String, Array[Int]) => Boolean

  protected def falseModelFilter(modelLoader: ModelLoader, objectType: String, objectName: String, dims: Array[Int]): Boolean = {
    // Skip these kinds of thing because they are likely a model of some kind.
    !(objectType == "#Parameter#" && objectName.matches(".*/_[0-9]+$"))
  }

  def loadExpressions(path: String, namespace: String = ""): Map[String, Expression] =
      filteredLoadExpressions(path, namespace, falseModelFilter)

  protected def filteredLoadExpressions(path: String, namespace: String = "", modelFilter: ModelFilter = falseModelFilter): Map[String, Expression] = {

    def read(line: String, modelLoader: ModelLoader, pc: ParameterCollection): Option[(String, Expression)] = {
      val Array(objectType, objectName, dimension, _, _) = line.split(" ")
      // Skip leading { and trailing }
      val dims = dimension.substring(1, dimension.length - 1).split(",").map(_.toInt)

      if (objectName.startsWith(namespace) && modelFilter(modelLoader, objectType, objectName, dims)) {
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

    val expressions = new ClosableModelLoader(path).autoClose { modelLoader =>
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

  // More of these will be added, plus design will change if the format can be generalized.
  def loadLstm(path: String, namespace: String = ""): (Option[LstmBuilder], Map[String, Expression]) = ???
  def loadCoupledLstm(path: String, namespace: String = ""): (Option[CoupledLstmBuilder], Map[String, Expression]) = ???

  def loadVanillaLstm(path: String, namespace: String = ""): (Option[VanillaLstmBuilder], Map[String, Expression]) = {
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

    // Could throw exception rather than use option or turn this into a collection.
    val expressions = filteredLoadExpressions(path, namespace, lstmModelFilter)
    val lstmBuilder =
        if (layers >= 2)
          Option(new VanillaLstmBuilder(layers - 2, inputDim, hiddenDim, model))
        else
          None

    (lstmBuilder, expressions)
  }

  val W2V_SIZE = 1234
  val VOC_SIZE = 3671

  val WEM_DIMENSIONS = 100
  val NUM_LAYERS = 1
  val FF_HIDDEN_DIM = 10
  val HIDDEN_DIM = 20

  val W_KEY = "/W"
  val B_KEY = "/b"
  val V_KEY = "/V"
  val W2V_WEMB_KEY = "/w2v-wemb"
  val MISSING_WEB_KEY = "/missing-wemb"

  def write(filename: String): Unit = {
    val pc = new ParameterCollection

    val W_p: Parameter = pc.addParameters(Dim(Seq(FF_HIDDEN_DIM, HIDDEN_DIM)))
    val b_p: Parameter = pc.addParameters(Dim(Seq(FF_HIDDEN_DIM)))
    val V_p: Parameter = pc.addParameters(Dim(Seq(1, FF_HIDDEN_DIM)))

    val w2v_wemb_lp: LookupParameter = pc.addLookupParameters(W2V_SIZE, Dim(Seq(WEM_DIMENSIONS)))
    val missing_wemb_lp: LookupParameter = pc.addLookupParameters(VOC_SIZE, Dim(Seq(WEM_DIMENSIONS)))

    val model = new ParameterCollection
    /*val builder = */ new VanillaLstmBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, model)

    new ClosableModelSaver(filename).autoClose { saver =>
      saver.addParameter(W_p, W_KEY)
      saver.addParameter(b_p, B_KEY)
      saver.addParameter(V_p, V_KEY)
      saver.addLookupParameter(w2v_wemb_lp, W2V_WEMB_KEY)
      saver.addLookupParameter(missing_wemb_lp, MISSING_WEB_KEY)
      saver.addModel(model)
    }
  }

  // May have to have several versions to accounts for all builders
  def transduce(builder: RnnBuilder, inputs: Iterable[Expression]): Option[Expression] =
    inputs.foldLeft(None: Option[Expression]){ (_, input) => Some(builder.addInput(input)) }
  //        if (inputs.size == 0)
  //          None
  //        else {
  //          inputs.dropRight(1).foreach { builder.addInput(_) }
  //          Some(builder.addInput(inputs.last))
  //          }

  def read(filename: String): Unit = {
    val (optionBuilder, expressions) = loadVanillaLstm(filename)

    expressions.keys.foreach(println)

    val builder = optionBuilder.get
    val W = expressions(W_KEY)
    val b = expressions(B_KEY)
    val V = expressions(V_KEY)
    val w2v_wemb = expressions(W2V_WEMB_KEY)
    val missing_wemb = expressions(MISSING_WEB_KEY)

    // An example run...
    val inputs = 0.until(100) map { index =>
      if (index % 2 == 0)
        Expression.pick(w2v_wemb, index % W2V_SIZE, 1) // size implied from eventual dictionary
      else
        Expression.pick(missing_wemb, index % VOC_SIZE, 1) // size implied from eventual dictionary
    }

    builder.newGraph()
    builder.startNewSequence()

    val selected = transduce(builder, inputs).get
    val prediction = Expression.logistic(V * (W * selected + b))

    print(prediction.value().toSeq())
  }

  def main(args: Array[String]): Unit = {
    val filename = "model.dy.kwa"

    Initialize.initialize(Map("random-seed" -> 2522620396l))
    write(filename)
    read(filename)
  }
}
