package org.clulab.dynet

import edu.cmu.dynet._

object Utils extends App {

  def load(path:String, pc:ParameterCollection, nameSpace: String = "/"):Map[String, Any] = {

    abstract class MetaData
    case class ParamMetadata(name:String, dims:Dim) extends MetaData
    case class LookupParamMetadata(name:String, dims:Dim, numElems:Int) extends MetaData


    def readParams(lines:Iterator[String], pc:ParameterCollection):List[MetaData] = {
      if(lines.isEmpty)
        List.empty[MetaData]
      else{
        val header = lines.next()
        val Array(typ, name, dimensions, huh, grad) = header.split(" ")
        val dims = dimensions.substring(1, dimensions.length-1).split(",").map(_.toInt)

        if(typ == "#Parameter#"){
          ParamMetadata(name, Dim(dims))::readParams(lines, pc)
        }
        else if(typ == "#LookupParameter#") {
          val numElems = dims(1)
          LookupParamMetadata(name, Dim(dims(0)), numElems)::readParams(lines, pc)
        }
        else{
          throw new RuntimeException("Unrecognized line in model file")
        }
      }
    }

    val src = io.Source.fromFile(path)

    val lines = src.getLines() filter (_.startsWith("#")) filter (_.contains(nameSpace))

    val params = readParams(lines, pc)

    src.close()

    val loader = new ModelLoader(path)

    val ret = params map {
      case ParamMetadata(name, dim) =>
        val param = pc.addParameters(dim)
        loader.populateParameter(param, key = name)
        name -> param
      case LookupParamMetadata(name, dim, numElems) =>
        val param = pc.addLookupParameters(numElems, dim)
        loader.populateLookupParameter(param, key = name)
        name -> param
    }

    loader.done()

    ret.toMap
  }

  // Ad-hoc test
  val pc = new ParameterCollection
  val x = load("model.dy", pc, "/vanilla-lstm-builder/")
  println(x.size)
}
