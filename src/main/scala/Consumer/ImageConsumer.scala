package Consumer

// =====================================================================================
// CONSUMER (etapes 3, 4 et 7 du schema)
//
// Role : traiter les images en temps reel, au fur et a mesure qu'elles arrivent.
//   3. il surveille data/destination/ et detecte chaque nouveau fichier .png
//   7. il charge le modele ONNX exporte par le service ML (et le recharge s'il change)
//   4. il ecrit ses resultats dans data/output/ (lus par le service ML et par Streamlit)
//
// Technologie : Spark Structured Streaming en Scala.
// Le flux est traite en "micro-batchs" : toutes les 5 secondes (trigger-interval),
// Spark regarde quels fichiers sont nouveaux et les traite ensemble.
//
// Trois requetes de streaming tournent en parallele sur le meme flux d'images :
//   - "images"         : une ligne par image (caracteristiques + prediction) -> images.csv
//   - "stats_by_label" : statistiques cumulees par classe                    -> stats_by_label.csv
//   - "window_counts"  : nombre d'images par classe et par tranche de 10 s  -> window_counts.csv
//
// Parametres : section "consumer" de application.conf.
// =====================================================================================

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._                     // col, avg, count, window, regexp_extract...
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types._                         // types du schema (StringType, BinaryType...)
import com.typesafe.config.ConfigFactory
import java.io.{ByteArrayInputStream, File, FileWriter, PrintWriter}
import java.nio.file.{AtomicMoveNotSupportedException, Files, StandardCopyOption}
import javax.imageio.ImageIO                                // decodage des images PNG (inclus dans Java)
import scala.concurrent.duration.Duration

// caracteristiques extraites de chaque image (partie C)
// Ce sont ces nombres que le modele utilise pour reconnaitre la classe : il ne "voit" pas l'image,
// seulement ce resume chiffre.
case class ImageFeatures(
  width: Int,               // largeur en pixels
  height: Int,              // hauteur en pixels
  meanIntensity: Double,    // luminosite moyenne (0 = noir, 255 = blanc)
  stdIntensity: Double,     // contraste : ecart-type des niveaux de gris
  entropy: Double,          // desordre de l'histogramme des niveaux de gris
  gradient: Double,         // difference moyenne entre pixels voisins (texture)
  histogram: Array[Double], // HistBins classes de niveaux de gris, normalisees
  // texture : matrice de co-occurrence (GLCM) des niveaux de gris de deux pixels voisins
  glcmContrast: Double,     // grand si les voisins ont des niveaux tres differents
  glcmHomogeneity: Double,  // grand si les voisins ont des niveaux proches (zones lisses)
  glcmEnergy: Double,       // grand si peu de couples de niveaux differents (texture reguliere)
  glcmCorrelation: Double,  // lien lineaire entre le niveau d'un pixel et celui de son voisin
  lbp: Array[Double],       // LbpBins motifs locaux (LBP) : bords, coins, zones plates...
  // zone centrale de l'image (moitie de la largeur et de la hauteur), ou se trouve souvent la lesion
  centerMean: Double,       // luminosite moyenne du centre
  centerStd: Double,        // contraste du centre
  centerRatio: Double       // luminosite du centre / luminosite de toute l'image
)

object ImageConsumer {

  // l'histogramme regroupe les 256 niveaux de gris en 16 classes de 16 niveaux
  val HistBins = 16
  // noms des colonnes de l'histogramme : h00, h01, ..., h15
  val histColumns: Seq[String] = (0 until HistBins).map(i => f"h$i%02d")
  // la GLCM compare les niveaux de gris regroupes en 16 classes (256 niveaux seraient trop fins)
  val GlcmLevels = 16
  // LBP "uniforme invariant en rotation" a 8 voisins : 0 a 8 voisins plus clairs que le centre
  // (motifs simples), plus une classe 9 pour tous les motifs irreguliers
  val LbpBins = 10
  val lbpColumns: Seq[String] = (0 until LbpBins).map(i => s"lbp$i")
  val textureColumns: Seq[String] =
    Seq("glcmContrast", "glcmHomogeneity", "glcmEnergy", "glcmCorrelation") ++ lbpColumns ++
      Seq("centerMean", "centerStd", "centerRatio")

  // colonnes utilisables par le modele (le service ML choisit parmi celles-ci)
  val featureColumns: Seq[String] =
    Seq("width", "height", "meanIntensity", "stdIntensity", "entropy", "gradient") ++ histColumns ++ textureColumns

  // schema fixe du format binaryFile (obligatoire pour une source fichier en streaming)
  // Pour chaque fichier, Spark fournit : son chemin, sa date de modification,
  // sa taille en octets et son contenu brut (les octets du PNG).
  val binarySchema: StructType = StructType(Seq(
    StructField("path", StringType),
    StructField("modificationTime", TimestampType),
    StructField("length", LongType),
    StructField("content", BinaryType)
  ))

  // decode le PNG et calcule les caracteristiques sur l'image en niveaux de gris
  // Renvoie None si les octets ne sont pas une image lisible.
  def extractFeatures(bytes: Array[Byte]): Option[ImageFeatures] = {
    val img = ImageIO.read(new ByteArrayInputStream(bytes))
    if (img == null) return None

    val w = img.getWidth
    val h = img.getHeight
    // tous les pixels d'un coup, ligne par ligne ; chaque pixel = un entier qui contient R, G et B
    val pixels = img.getRGB(0, 0, w, h, null, 0, w)
    val n = pixels.length.toDouble

    val gray = new Array[Int](pixels.length)   // niveau de gris de chaque pixel
    val hist256 = new Array[Long](256)         // combien de pixels pour chaque niveau 0..255
    var sum = 0.0
    var sumSq = 0.0
    // un tour par pixel (i = position du pixel dans le tableau)
    for (i <- pixels.indices) {
      val p = pixels(i)
      // extraction de R (bits 16-23), G (bits 8-15), B (bits 0-7),
      // puis conversion en gris avec les poids standards de la perception humaine
      val g = math.min(255, (0.299 * ((p >> 16) & 0xff) + 0.587 * ((p >> 8) & 0xff) + 0.114 * (p & 0xff)).toInt)
      gray(i) = g
      hist256(g) += 1
      sum += g
      sumSq += g.toDouble * g
    }
    val mean = sum / n
    // ecart-type = racine(moyenne des carres - carre de la moyenne)
    val std = math.sqrt(math.max(sumSq / n - mean * mean, 0.0))

    // entropie de Shannon (en bits) : faible si l'image a peu de niveaux de gris differents,
    // elevee si les niveaux sont tres varies
    val entropy = hist256.filter(_ > 0).map { c =>
      val pr = c / n
      -pr * math.log(pr) / math.log(2)
    }.sum

    // voisins de droite et du dessous
    // gradient moyen : grand si l'image a beaucoup de contours / de grain, petit si elle est lisse
    // glcm(a)(b) : nombre de couples de voisins dont l'un est au niveau a et l'autre au niveau b
    // (comptes dans les deux sens : la matrice est symetrique)
    val glcm = Array.ofDim[Double](GlcmLevels, GlcmLevels)
    var gradSum = 0.0
    var gradCount = 0L
    for (y <- 0 until h; x <- 0 until w) {
      val idx = y * w + x   // position du pixel (x, y) dans le tableau a une dimension
      val a = gray(idx) * GlcmLevels / 256
      if (x + 1 < w) {
        gradSum += math.abs(gray(idx) - gray(idx + 1)); gradCount += 1
        val b = gray(idx + 1) * GlcmLevels / 256
        glcm(a)(b) += 1; glcm(b)(a) += 1
      }
      if (y + 1 < h) {
        gradSum += math.abs(gray(idx) - gray(idx + w)); gradCount += 1
        val b = gray(idx + w) * GlcmLevels / 256
        glcm(a)(b) += 1; glcm(b)(a) += 1
      }
    }
    val gradient = if (gradCount > 0) gradSum / gradCount else 0.0

    // GLCM en proportions (la somme vaut 1), puis ses 4 indicateurs classiques (Haralick)
    val glcmTotal = math.max(glcm.map(_.sum).sum, 1.0)
    val cells = for (i <- 0 until GlcmLevels; j <- 0 until GlcmLevels) yield (i, j, glcm(i)(j) / glcmTotal)
    val glcmMean = cells.map { case (i, _, p) => i * p }.sum
    val glcmVar = cells.map { case (i, _, p) => (i - glcmMean) * (i - glcmMean) * p }.sum
    val glcmContrast = cells.map { case (i, j, p) => (i - j) * (i - j) * p }.sum
    val glcmHomogeneity = cells.map { case (i, j, p) => p / (1 + math.abs(i - j)) }.sum
    val glcmEnergy = cells.map { case (_, _, p) => p * p }.sum
    val glcmCorrelation =
      if (glcmVar > 0) cells.map { case (i, j, p) => (i - glcmMean) * (j - glcmMean) * p }.sum / glcmVar else 0.0

    // LBP : pour chaque pixel (hors bords), on regarde ses 8 voisins dans l'ordre du tour
    // et on note lesquels sont au moins aussi clairs que lui. Si ce tour change au plus 2 fois
    // entre "plus clair" et "plus sombre", le motif est simple (bord, coin, point...) : on le
    // classe selon le nombre de voisins clairs (0 a 8). Sinon c'est un motif irregulier (classe 9).
    val neighbours = Seq((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    val lbpHist = new Array[Double](LbpBins)
    for (y <- 1 until h - 1; x <- 1 until w - 1) {
      val c = gray(y * w + x)
      val bits = neighbours.map { case (dy, dx) => if (gray((y + dy) * w + x + dx) >= c) 1 else 0 }
      val transitions = bits.indices.count(k => bits(k) != bits((k + 7) % 8))
      lbpHist(if (transitions <= 2) bits.sum else LbpBins - 1) += 1
    }
    val lbpTotal = math.max(lbpHist.sum, 1.0)
    val lbp = lbpHist.map(_ / lbpTotal)

    // zone centrale : on retire un quart de la largeur et de la hauteur de chaque cote
    val centerPixels = for (y <- h / 4 until h - h / 4; x <- w / 4 until w - w / 4) yield gray(y * w + x).toDouble
    val centerMean = if (centerPixels.nonEmpty) centerPixels.sum / centerPixels.length else mean
    val centerStd =
      if (centerPixels.nonEmpty)
        math.sqrt(math.max(centerPixels.map(v => v * v).sum / centerPixels.length - centerMean * centerMean, 0.0))
      else std
    val centerRatio = if (mean > 0) centerMean / mean else 0.0

    // histogramme reduit a 16 classes, en proportion de pixels (la somme vaut 1) :
    // ainsi deux images de tailles differentes restent comparables
    val binWidth = 256 / HistBins
    val histogram = (0 until HistBins).map { b =>
      hist256.slice(b * binWidth, (b + 1) * binWidth).sum / n
    }.toArray

    Some(ImageFeatures(w, h, mean, std, entropy, gradient, histogram,
      glcmContrast, glcmHomogeneity, glcmEnergy, glcmCorrelation, lbp, centerMean, centerStd, centerRatio))
  }

  def main(args: Array[String]): Unit = {

    // "spark.sql.shuffle.partitions" = 4 : nombre de partitions pour les agregations (groupBy).
    // La valeur par defaut (200) est faite pour de gros volumes : ici elle ralentirait tout.
    val spark = SparkSession.builder()
      .appName("BUSI Image Consumer")
      .master("local[*]")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    //recuperer de .conf
    val cfg           = ConfigFactory.load().getConfig("consumer")
    val inputDir       = cfg.getString("input-dir")          // dossier surveille
    val outputDir      = cfg.getString("output-dir")         // ou ecrire les CSV de resultats
    val checkpointDir  = cfg.getString("checkpoint-dir")     // memoire de Spark (voir plus bas)
    val modelDir       = cfg.getString("model-dir")          // ou trouver busi_model.onnx
    val triggerEvery   = Duration(cfg.getString("trigger-interval"))  // frequence des micro-batchs
    val maxFiles       = cfg.getInt("max-files-per-trigger") // nb max d'images par micro-batch
    val windowDuration = cfg.getString("window-duration")    // taille des fenetres de temps
    val watermark      = cfg.getString("watermark")          // retard max accepte (voir plus bas)

    new File(outputDir).mkdirs()
    println(s"input=$inputDir output=$outputDir checkpoint=$checkpointDir model=$modelDir trigger=$triggerEvery")

    // modele ONNX exporte par le service ML, recharge des qu'il change
    // (au demarrage le fichier peut ne pas exister : les predictions restent alors vides)
    val model = new OnnxModel(new File(modelDir, "busi_model.onnx"))

    // UDF = "User Defined Function" : permet d'appeler notre fonction Scala extractFeatures
    // sur une colonne Spark. Elle sera executee pour chaque image du flux.
    // asNondeterministic : evite que Spark rappelle l'UDF pour chaque champ de "features.*"
    val featuresUdf = udf((bytes: Array[Byte]) => extractFeatures(bytes)).asNondeterministic()

    // ---------- A. Ingestion ----------
    // readStream (et non read) = lecture en continu : Spark relit le dossier a chaque micro-batch
    // et ne traite que les fichiers qu'il n'a pas encore vus.
    // Les fichiers caches (commencant par "." ou "_") sont ignores automatiquement.
    val raw = spark.readStream
      .format("binaryFile")
      .schema(binarySchema)
      .option("maxFilesPerTrigger", maxFiles)
      .load(s"$inputDir/*.png")

    // le label est dans le nom du fichier : "benign (3).png" -> benign
    // (pour une image envoyee depuis Streamlit, "upload_...png", le label reste vide)
    val images = raw
      // nom du fichier = tout ce qui suit le dernier "/" du chemin
      .withColumn("fileName", regexp_extract(col("path"), "[^/]+$", 0))
      .withColumn("label", regexp_extract(col("fileName"), "^(benign|malignant|normal)", 1))
      // calcul des caracteristiques : une colonne "features" qui contient une structure ImageFeatures
      .withColumn("features", featuresUdf(col("content")))
      // heure a laquelle Spark a traite l'image
      .withColumn("ingestionTime", current_timestamp())
      // on "aplatit" la structure features en colonnes simples, et on abandonne "content"
      // (les octets de l'image : inutile de les garder une fois les caracteristiques calculees)
      .select(
        Seq(
          col("fileName"),
          col("label"),
          col("length").as("sizeBytes"),
          col("modificationTime"),
          col("ingestionTime"),
          col("features.width").as("width"),
          col("features.height").as("height"),
          round(col("features.meanIntensity"), 2).as("meanIntensity"),
          round(col("features.stdIntensity"), 2).as("stdIntensity"),
          round(col("features.entropy"), 4).as("entropy"),
          round(col("features.gradient"), 4).as("gradient")
        ) ++ histColumns.zipWithIndex.map { case (name, i) =>
          // une colonne par case de l'histogramme : h00 = features.histogram[0], etc.
          round(col("features.histogram")(i), 5).as(name)
        } ++ Seq("glcmContrast", "glcmHomogeneity", "glcmEnergy", "glcmCorrelation").map { name =>
          round(col(s"features.$name"), 5).as(name)
        } ++ lbpColumns.zipWithIndex.map { case (name, i) =>
          round(col("features.lbp")(i), 5).as(name)
        } ++ Seq(
          round(col("features.centerMean"), 2).as("centerMean"),
          round(col("features.centerStd"), 2).as("centerStd"),
          round(col("features.centerRatio"), 4).as("centerRatio")
        ): _*   // ": _*" passe la liste de colonnes comme arguments separes a select(...)
      )

    // ---------- D. Prediction ----------
    // une ligne par image (caracteristiques + prediction), ajoutee a images.csv
    // foreachBatch : a chaque micro-batch, Spark nous donne les nouvelles lignes sous forme
    // de DataFrame classique (batchDF) et le numero du batch (batchId). On en fait ce qu'on veut.
    val writeImages: (DataFrame, Long) => Unit = (batchDF, batchId) => {
      // collect() ramene les lignes sur le driver : correct ici car il n'y a que quelques images par batch
      val rows = batchDF.collect()
      if (rows.nonEmpty) {
        // si le service ML a exporte un nouveau modele depuis le dernier batch, on le charge
        model.refresh()
        // chaque ligne est transformee en Map "nom de colonne -> valeur" pour le modele
        val predictions = model.predict(rows.map(r => r.getValuesMap[Any](batchDF.columns)))
        // on ajoute a chaque ligne : classe predite, confiance, version du modele, numero du batch
        val outRows = rows.zip(predictions).map { case (r, p) =>
          val predCols: Seq[Any] = p match {
            case Some(pred) => Seq(pred.label, pred.confidence, pred.version)
            case None       => Seq(null, null, null) // pas encore de modele
          }
          (r.toSeq ++ predCols) :+ batchId
        }
        val header = batchDF.columns.toSeq ++ Seq("predictedLabel", "confidence", "modelVersion", "batchId")
        appendCsv(new File(outputDir, "images.csv"), header, outRows)
        println(s"[images] batch $batchId : ${rows.length} image(s), modele=${model.version.getOrElse("aucun")}")
      }
    }

    // demarrage de la requete "images"
    //  - outputMode "append" : on ne recoit que les nouvelles lignes a chaque batch
    //  - checkpointLocation : Spark y note quels fichiers il a deja traites. Au redemarrage,
    //    il reprend la ou il s'etait arrete au lieu de tout retraiter.
    //  - trigger : un micro-batch toutes les triggerEvery (5 s)
    val imagesQuery = images.writeStream
      .queryName("images")
      .outputMode("append")
      .foreachBatch(writeImages)
      .option("checkpointLocation", s"$checkpointDir/images")
      .trigger(Trigger.ProcessingTime(triggerEvery.toMillis))
      .start()

    // ---------- B. Agregations ----------
    // statistiques cumulees par classe (mode complete : table entiere a chaque batch)
    // Spark garde en memoire les totaux depuis le debut ("etat") et les met a jour a chaque batch.
    val statsByLabel = images
      .groupBy("label")
      .agg(
        count("*").as("nbImages"),
        round(avg("sizeBytes") / 1024, 1).as("avgSizeKB"),
        round(avg("width"), 1).as("avgWidth"),
        round(avg("height"), 1).as("avgHeight"),
        round(avg("meanIntensity"), 2).as("avgIntensity"),
        round(avg("stdIntensity"), 2).as("avgContrast"),
        round(avg("entropy"), 3).as("avgEntropy"),
        round(avg("gradient"), 3).as("avgGradient")
      )

    // a chaque batch on recoit la table complete (3 lignes, une par classe) : on remplace le fichier
    val writeStats: (DataFrame, Long) => Unit = (batchDF, _) => {
      val df = batchDF.orderBy("label")
      overwriteCsv(new File(outputDir, "stats_by_label.csv"), df.columns, df.collect().map(_.toSeq))
    }

    val statsQuery = statsByLabel.writeStream
      .queryName("stats_by_label")
      .outputMode("complete")
      .foreachBatch(writeStats)
      .option("checkpointLocation", s"$checkpointDir/stats_by_label")
      .trigger(Trigger.ProcessingTime(triggerEvery.toMillis))
      .start()

    // nombre d'images par classe et par fenetre de temps (heure d'arrivee du fichier)
    // window(...) range chaque image dans une tranche de 10 s selon sa date de modification
    // (ex : 14:30:00-14:30:10). Le watermark (30 s) dit a Spark qu'une image ne peut pas
    // arriver avec plus de 30 s de retard : il peut donc oublier les vieilles fenetres.
    val windowCounts = images
      .withWatermark("modificationTime", watermark)
      .groupBy(window(col("modificationTime"), windowDuration), col("label"))
      .agg(count("*").as("nbImages"))
      .select(
        col("window.start").as("windowStart"),
        col("window.end").as("windowEnd"),
        col("label"),
        col("nbImages")
      )

    val writeWindows: (DataFrame, Long) => Unit = (batchDF, _) => {
      val df = batchDF.orderBy("windowStart", "label")
      overwriteCsv(new File(outputDir, "window_counts.csv"), df.columns, df.collect().map(_.toSeq))
    }

    val windowQuery = windowCounts.writeStream
      .queryName("window_counts")
      .outputMode("complete")
      .foreachBatch(writeWindows)
      .option("checkpointLocation", s"$checkpointDir/window_counts")
      .trigger(Trigger.ProcessingTime(triggerEvery.toMillis))
      .start()

    println(s"Requetes demarrees : ${Seq(imagesQuery, statsQuery, windowQuery).map(_.name).mkString(", ")}")
    // le programme reste bloque ici et le streaming tourne indefiniment
    // (jusqu'a l'arret du programme ou une erreur dans une des requetes)
    spark.streams.awaitAnyTermination()
    model.close()
  }

  // ---------- ecriture CSV (petits volumes, cote driver) ----------
  // On ecrit les CSV nous-memes plutot qu'avec le writer CSV de Spark :
  // Spark ecrirait un dossier rempli de fichiers "part-0000...", moins pratique a lire pour Streamlit.

  // transforme une ligne en texte CSV : les textes sont entre guillemets, les valeurs nulles sont vides
  private def csvLine(values: Seq[Any]): String = values.map {
    case null      => ""
    case s: String => "\"" + s.replace("\"", "\"\"") + "\""
    case v         => v.toString
  }.mkString(",")

  // ajoute des lignes a la fin du fichier (et ecrit l'en-tete si le fichier n'existe pas encore)
  private def appendCsv(file: File, header: Seq[String], rows: Seq[Seq[Any]]): Unit = {
    val isNew = !file.exists()
    val out = new PrintWriter(new FileWriter(file, true))   // true = mode ajout
    try {
      if (isNew) out.println(header.mkString(","))
      rows.foreach(r => out.println(csvLine(r)))
    } finally out.close()
  }

  // ecrit dans un fichier temporaire puis remplace : Streamlit ne lit jamais un fichier a moitie ecrit
  private def overwriteCsv(file: File, header: Seq[String], rows: Seq[Seq[Any]]): Unit = {
    val tmp = new File(file.getParentFile, s".${file.getName}.tmp")
    val out = new PrintWriter(new FileWriter(tmp, false))   // false = on ecrase
    try {
      out.println(header.mkString(","))
      rows.foreach(r => out.println(csvLine(r)))
    } finally out.close()
    // remplacement atomique si le systeme le permet, sinon remplacement simple
    try Files.move(tmp.toPath, file.toPath, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    catch {
      case _: AtomicMoveNotSupportedException =>
        Files.move(tmp.toPath, file.toPath, StandardCopyOption.REPLACE_EXISTING)
    }
  }
}
