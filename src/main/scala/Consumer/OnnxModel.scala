package Consumer

// =====================================================================================
// CHARGEMENT ET UTILISATION DU MODELE (etape 7 du schema)
//
// Le service ML (Python) entraine un modele et l'exporte au format ONNX dans model/busi_model.onnx.
// ONNX est un format standard : un modele entraine en Python peut etre execute en Java/Scala
// grace a la librairie ONNX Runtime (dependance "onnxruntime" dans build.sbt).
//
// Cette classe :
//   - charge le fichier .onnx, et le recharge automatiquement quand Python en exporte un nouveau
//   - fait les predictions pour les images d'un micro-batch
// =====================================================================================

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import java.io.File
import java.nio.file.Files
import java.util.Collections
import scala.collection.JavaConverters._   // convertit les Map Java en Map Scala (.asScala)

// resultat pour une image : classe predite, confiance (probabilite de cette classe) et version du modele
case class Prediction(label: String, confidence: Double, version: String)

// modele ONNX exporte par le service ML (Python) ; recharge a chaud quand le fichier change.
// Les metadonnees du modele donnent l'ordre des colonnes ("features"), les classes ("labels") et la "version".
class OnnxModel(file: File) {

  // environnement ONNX Runtime (un seul par programme)
  private val env = OrtEnvironment.getEnvironment()
  // session = le modele charge en memoire, pret a predire (None tant qu'aucun modele n'existe)
  private var session: Option[OrtSession] = None
  // date de modification du fichier charge : permet de savoir s'il a change depuis
  private var loadedAt = 0L
  // noms des classes dans l'ordre des sorties du modele : benign, malignant, normal
  private var labels: Array[String] = Array.empty
  // noms des colonnes dans l'ordre attendu par le modele : width, height, meanIntensity...
  private var features: Array[String] = Array.empty
  var version: Option[String] = None

  // appele a chaque micro-batch : ne fait rien si le fichier n'existe pas ou n'a pas change
  def refresh(): Unit = {
    if (!file.exists() || file.lastModified() == loadedAt) return
    val lastModified = file.lastModified()
    var newSession: OrtSession = null
    try {
      // chargement depuis les octets : le fichier n'est pas verrouille (Python peut le remplacer)
      newSession = env.createSession(Files.readAllBytes(file.toPath), new OrtSession.SessionOptions())
      // metadonnees ajoutees par train_model.py au moment de l'export
      val meta = newSession.getMetadata.getCustomMetadata.asScala
      val newLabels = meta("labels").split(",")
      val newFeatures = meta("features").split(",")

      // le nouveau modele est valide : on remplace l'ancien (et on libere sa memoire)
      session.foreach(_.close())
      session = Some(newSession)
      labels = newLabels
      features = newFeatures
      version = meta.get("version")
      loadedAt = lastModified
      println(s"[modele] version ${version.getOrElse("?")} chargee : ${labels.mkString("/")}, ${features.length} caracteristiques")
    } catch {
      // fichier en cours d'ecriture, corrompu ou sans metadonnees : on garde l'ancien modele
      // et on reessaiera au prochain batch (loadedAt n'a pas change)
      case e: Exception =>
        if (newSession != null) newSession.close()
        println(s"[modele] chargement impossible (${e.getMessage}), nouvel essai au prochain batch")
    }
  }

  // une prediction par ligne ; None si pas de modele ou si une caracteristique manque
  // rows : une Map "nom de colonne -> valeur" par image
  def predict(rows: Seq[Map[String, Any]]): Seq[Option[Prediction]] = session match {
    case None => rows.map(_ => None)
    case Some(s) =>
      // pour chaque image, on construit le vecteur de nombres dans l'ordre exact attendu par le modele
      val vectors = rows.map(r => features.map(f => toFloat(r.getOrElse(f, null))))
      // on ecarte les images dont une valeur manque (NaN), par exemple une image illisible
      val validIdx = vectors.indices.filter(i => !vectors(i).exists(_.isNaN))
      if (validIdx.isEmpty) return rows.map(_ => None)

      // tenseur = tableau de nombres a 2 dimensions [nombre d'images, nombre de caracteristiques]
      // On predit toutes les images du batch en un seul appel.
      val tensor = OnnxTensor.createTensor(env, validIdx.map(vectors(_)).toArray)
      val result = s.run(Collections.singletonMap(s.getInputNames.iterator.next, tensor))
      try {
        // sorties skl2onnx : 0 = label, 1 = probabilites [n, nbClasses]
        // ex. pour une image : [0.20, 0.65, 0.15] -> 65 % malignant
        val probs = result.get(1).getValue.asInstanceOf[Array[Array[Float]]]
        val byRow = validIdx.zip(probs.map { p =>
          // la classe predite est celle qui a la plus forte probabilite
          val best = p.indices.maxBy(p(_))
          // confiance arrondie a 3 decimales
          Prediction(labels(best), math.round(p(best) * 1000) / 1000.0, version.orNull)
        }).toMap
        // on remet les predictions dans l'ordre des lignes (None pour les images ecartees)
        rows.indices.map(byRow.get)
      } finally {
        // les tenseurs ONNX utilisent de la memoire hors de Java : il faut les liberer a la main
        result.close()
        tensor.close()
      }
  }

  def close(): Unit = session.foreach(_.close())

  // le modele attend des Float : on convertit Int/Double/Long, et une valeur absente devient NaN
  private def toFloat(v: Any): Float = v match {
    case n: Number => n.floatValue()
    case _         => Float.NaN
  }
}
