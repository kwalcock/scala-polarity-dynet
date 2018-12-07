package org.clulab.scalaPolarityDynet

import edu.cmu.dynet._
import org.clulab.fatdynet.utils.Closer.AutoCloser
import org.clulab.fatdynet.utils.Loader.{loadVanillaLstm, transduce}

import scala.io.Source

object ReaderApp extends App{

  def run_instance(words: Seq[String], builder: VanillaLstmBuilder, /*missing_wemb: Expression,*/ w2v_wemb: Expression,
      missing_voc: Map[String, Int], w2v_voc: Map[String, Int], W: Expression, V: Expression, b: Expression): Expression = {

    val inputs = words.flatMap { unsanitizedWord =>
      val word = sanitizeWord(unsanitizedWord, keepNumbers = false)

      if (w2v_voc.contains(word))
        Some(Expression.pick(w2v_wemb, w2v_voc(word) % 11691, 1)) // TODO: model files have been abbreviated
//      else if (missing_voc.contains(word))
//        Some(Expression.pick(missing_wemb, missing_voc(word) % 100, 1)) // TODO: model files have been abbreviated, so % 100
      else
        None
    }

    builder.newGraph()
    builder.startNewSequence()

    val selected = transduce(builder, inputs).get
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

  val p = "model.dy"
  val dictPath = "vocab.txt"
  val w2vDictPath = "w2vvoc.txt"
  val missing_voc = Source.fromFile(dictPath).autoClose { source =>
    source
        .getLines()
        .zipWithIndex
        .toMap
  }
  val w2v_voc = Source.fromFile(w2vDictPath).autoClose { source =>
    source
        .getLines()
        .zipWithIndex
        .toMap
  }

  Initialize.initialize(Map("random-seed" -> 2522620396l))

  val (optionBuilder, expressions) = loadVanillaLstm(p)
  val W = expressions("/W")
  val b = expressions("/b")
  val V = expressions("/V")
  val w2v_wemb = expressions("/wemb")
//  val missing_wemb = expressions("/missing-wemb")

  val sentence = "To formally prove that increased ROS levels enhance anti-tumour effects of the SG-free diet , the authors crossed Emu-Myc mice with mice deficient for Tigar , a fructose-2 ,6-bisphosphatase , which limits glycolysis and favours pentose phosphate pathways , thus limiting ROS levels XREF_BIBR , XREF_BIBR ( XREF_FIG ) .".toLowerCase()
  val prediction = run_instance(sentence.split(" "), optionBuilder.get, /*missing_wemb,*/ w2v_wemb, missing_voc, w2v_voc, W, V, b)

  print(prediction.value().toSeq())
}
