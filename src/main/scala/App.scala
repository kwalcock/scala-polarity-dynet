import edu.cmu.dynet._


import scala.io.Source

object App extends App{
  Initialize.initialize(Map("random-seed" -> 2522620396l))
  val p = "/Users/enrique/scratch/lstm_polarity/model.dy"
  val dictPath = "/Users/enrique/scratch/lstm_polarity/vocab.txt"
  val w2vDictPath = "/Users/enrique/scratch/lstm_polarity/w2vvoc.txt"
  val pc = new ParameterCollection
  val lpc = new ParameterCollection

  val lines = Source.fromFile(dictPath).getLines().toList
  val lines2 = Source.fromFile(w2vDictPath).getLines().toList

  val missing_voc = lines.zipWithIndex.toMap
  val w2v_voc = lines2.zipWithIndex.toMap

  val VOC_SIZE = 3671//voc.size
  val WEM_DIMENSIONS = 100

  val NUM_LAYERS = 1
  val HIDDEN_DIM = 20

  val FF_HIDDEN_DIM = 10

  val missing_wemb_p = pc.addLookupParameters(VOC_SIZE, Dim(Seq(WEM_DIMENSIONS)))
  val w2v_wemb_p = pc.addLookupParameters(1579375, Dim(Seq(WEM_DIMENSIONS)))
  // Feed-Forward parameters
  val W_p = pc.addParameters(Dim(Seq(FF_HIDDEN_DIM, HIDDEN_DIM)))
  val b_p = pc.addParameters(Dim(Seq(FF_HIDDEN_DIM)))
  val V_p = pc.addParameters(Dim(Seq(1, FF_HIDDEN_DIM)))

  val builder = new LstmBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, lpc)

  val loader = new ModelLoader(p)

  //loader.populateModel(pc, "/")
  loader.populateModel(lpc, "/vanilla-lstm-builder/")

  loader.populateParameter(W_p, key = "/W")
  loader.populateParameter(b_p, key = "/b")
  loader.populateParameter(V_p, key = "/V")
  loader.populateLookupParameter(missing_wemb_p, key = "/missing-wemb")
  loader.populateLookupParameter(w2v_wemb_p, key = "/w2v-wemb")


  val W = Expression.parameter(W_p)
  val b = Expression.parameter(b_p)
  val V = Expression.parameter(V_p)
  val missing_wemb = Expression.parameter(missing_wemb_p)
  val w2v_wemb = Expression.parameter(w2v_wemb_p)



  def run_instance(instance:Seq[String], builder:LstmBuilder, missing_wemb:LookupParameter, w2v_wemb:LookupParameter, missing_voc:Map[String, Int], w2v_voc:Map[String, Int], W:Expression, V:Expression, b:Expression)  ={

    // Fetch the embeddings for the current sentence
    val words = instance
    val inputs = words map { w =>
      val word = sanitizeWord(w, keepNumbers = false)

      w2v_voc.contains(word) match {
        case true =>
          Expression.lookup(w2v_wemb, w2v_voc(word))
        case false =>
          Expression.lookup(missing_wemb, missing_voc(word))
      }
    }

    // Run FF over the LSTM
    builder.newGraph()
    builder.startNewSequence()
    val states = inputs map {
      w => builder.addInput(w)
    }


    // Get the last embedding
    val selected = states.last

    // Run the FF network for classification
    val prediction = Expression.logistic(V * (W * selected + b))

    prediction
  }

  def sanitizeWord(uw:String, keepNumbers:Boolean = true):String = {
    val w = uw.toLowerCase()

    // skip parens from corenlp
    if(w == "-lrb-" || w == "-rrb-" || w == "-lsb-" || w == "-rsb-") {
      return ""
    }

    // skip URLS
    if(w.startsWith("http") || w.contains(".com") || w.contains(".org")) //added .com and .org to cover more urls (becky)
      return ""

    // normalize numbers to a unique token
    if(isNumber(w)) {
      if(keepNumbers) return "xnumx"
      else return ""
    }

    // remove all non-letters; convert letters to lowercase
    val os = new collection.mutable.StringBuilder()
    var i = 0
    while(i < w.length) {
      val c = w.charAt(i)
      // added underscore since it is our delimiter for dependency stuff...
      if(Character.isLetter(c) || c == '_') os += c
      i += 1
    }
    os.toString()
  }

  def isNumber(w:String):Boolean = {
    var i = 0
    var foundDigit = false
    while(i < w.length) {
      val c = w.charAt(i)
      if(! Character.isDigit(c) &&
        c != '-' && c != '+' &&
        c != ',' && c != '.' &&
        c != '/' && c != '\\')
        return false
      if(Character.isDigit(c))
        foundDigit = true
      i += 1
    }
    foundDigit
  }

  val sentence = "To formally prove that increased ROS levels enhance anti-tumour effects of the SG-free diet , the authors crossed Emu-Myc mice with mice deficient for Tigar , a fructose-2 ,6-bisphosphatase , which limits glycolysis and favours pentose phosphate pathways , thus limiting ROS levels XREF_BIBR , XREF_BIBR ( XREF_FIG ) .".toLowerCase()


  val prediction = run_instance(sentence.split(" "), builder, missing_wemb_p,w2v_wemb_p, missing_voc, w2v_voc, W, V, b)

  print(prediction.value().toSeq())
}
