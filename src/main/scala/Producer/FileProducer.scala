package Producer

// =====================================================================================
// PRODUCER (etapes 1 et 2 du schema)
//
// Role : simuler l'arrivee d'echographies au fil de l'eau.
//   1. il lit toutes les images de data/Source/{benign,malignant,normal}/
//   2. il les copie par petits lots (batch-size images) dans data/destination/,
//      avec une pause (interval) entre deux lots.
// Le consumer surveille data/destination/ : chaque image deposee est donc,
// pour lui, une "nouvelle donnee" qui arrive dans le flux.
//
// Technologie : Spark Core (RDD) en Scala.
// Tous les parametres viennent de src/main/resources/application.conf (section "producer").
// =====================================================================================

import org.apache.spark.sql.SparkSession
import org.apache.hadoop.fs.{FileSystem, Path, FileUtil}   // API fichiers de Hadoop (utilisee par Spark)
import org.apache.hadoop.conf.Configuration
import com.typesafe.config.ConfigFactory                   // lecture de application.conf
import java.io.File
import scala.concurrent.duration.Duration                  // pour lire "2 seconds" depuis la conf

object FileProducer {

  def main(args: Array[String]): Unit = {

    // Point d'entree de Spark. "local[*]" = Spark tourne sur cette machine
    // et utilise tous les coeurs du processeur (pas de cluster).
    val spark = SparkSession.builder()
      .appName("BUSI File Producer")
      .master("local[*]")
      .getOrCreate()

    // SparkContext : l'objet "bas niveau" qui permet de manipuler des RDD
    val sc = spark.sparkContext

    // on n'affiche que les avertissements et erreurs de Spark (sinon il y a beaucoup de logs INFO)
    sc.setLogLevel("WARN")

    //recuperer de .conf
    val cfg           = ConfigFactory.load().getConfig("producer")
    val sourceDir      = cfg.getString("source-dir")       // ou lire les images
    val destinationDir = cfg.getString("destination-dir")  // ou les deposer (surveille par le consumer)
    val stagingDir     = cfg.getString("staging-dir")      // dossier tampon pour l'ecriture
    val batchSize      = cfg.getInt("batch-size")          // nombre d'images par lot
    val interval        = Duration(cfg.getString("interval"))  // pause entre deux lots
    val mode            = cfg.getString("mode")            // "copy" ou "move"
    val deleteSource    = mode == "move"                   // en mode "move" on supprime l'original
    val shuffle         = cfg.getBoolean("shuffle")        // melanger l'ordre d'envoi ?


    println(s"source=$sourceDir dest=$destinationDir batchSize=$batchSize interval=$interval mode=$mode")

    // recuperer le nombre d'image
    // binaryFiles lit tous les fichiers des sous-dossiers ("*/*" = un niveau de dossier par classe).
    // Il renvoie des paires (chemin, contenu) : on ne garde que le chemin,
    // on filtre les .png, puis collect() ramene la liste sur le driver (le programme principal).
    val files = sc.binaryFiles(s"$sourceDir/*/*")
      .map {
        case (path, _) => path
      }
      .filter ( _.toLowerCase.endsWith(".png"))
      // la base BUSI complete contient aussi les masques de segmentation ("benign (1)_mask.png") :
      // ce ne sont pas des echographies, le consumer les etiquetterait a tort comme "benign"
      .filter ( p => !p.toLowerCase.contains("_mask"))
      .collect()


    println(s"Nombre d'images trouvées : ${files.length}")


    // melanger pour que les classes arrivent dans le desordre, comme dans un vrai flux
    // (sinon toutes les "benign" arrivent d'abord et le modele ne voit les "normal" qu'a la fin)
    val ordered = if (shuffle) scala.util.Random.shuffle(files.toSeq) else files.toSeq

    //creer les batchs
    // grouped(2) decoupe la liste en paquets de 2 : [a,b,c,d,e] -> [a,b], [c,d], [e]
    val batches = ordered.grouped(batchSize).toSeq

    println(s"Nombre de batchs : ${batches.length}")

    //copier les batchs
    // zipWithIndex associe a chaque lot son numero (0, 1, 2...) pour l'affichage
    batches.zipWithIndex.foreach {

      case (batch, index) =>

        println(s"BATCH ${index + 1} ")

        // parallelize transforme la liste de chemins du lot en RDD :
        // Spark repartit les copies entre ses taches (ici, les coeurs de la machine)
        val rdd = sc.parallelize(batch)

        // foreachPartition : le code s'execute une fois par partition du RDD.
        // On ouvre donc le systeme de fichiers une seule fois par partition, pas une fois par image.
        rdd.foreachPartition { partition =>

          // preparation hadoop
          val conf = new Configuration()
          val fs = FileSystem.get(conf)   // systeme de fichiers local (ou HDFS sur un cluster)
          // pas de fichiers .crc a cote des images
          // (Hadoop ecrit par defaut un fichier de controle ".nom.png.crc" pour chaque fichier)
          fs.setWriteChecksum(false)

          partition.foreach { sourcePath =>
            val source = new Path(sourcePath)
            // meme nom de fichier, dans le dossier staging puis dans le dossier destination
            val staging = new Path(s"$stagingDir/${source.getName}")
            val destination = new Path(s"$destinationDir/${source.getName}")

            println(s"Copie : ${source.getName}")

            // 1) copie complete dans le dossier staging (non surveille par le consumer)
            //    La copie prend un peu de temps : si on ecrivait directement dans destination,
            //    le consumer pourrait lire une image incomplete.
            FileUtil.copy(
              fs,//systeme de fichier source
              source,
              fs,//systeme de fichier dest
              staging,
              deleteSource,//true si mode = "move"
              conf
            )
            // 2) rename atomique : le consumer ne voit jamais une image a moitie ecrite
            //    Un rename est instantane : le fichier apparait d'un coup, deja complet.
            //    Il echoue si une image du meme nom existe deja dans destination.
            fs.mkdirs(destination.getParent)
            if (!fs.rename(staging, destination))
              println(s"Echec du rename : ${source.getName}")
          }
          fs.close()
        }

        println(s"Batch ${index + 1} terminé.")
        // attendre le prochain
        // cette pause cree l'effet "flux" : les images arrivent petit a petit
        Thread.sleep(interval.toMillis)
    }
    // tous les lots sont envoyes : on arrete Spark proprement
    spark.stop()
  }

}
