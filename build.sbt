// Fichier de build sbt : version de Scala, dependances et options de lancement.

ThisBuild / version := "0.1.0-SNAPSHOT"

// Spark 3.5 est compile pour Scala 2.12 et 2.13 : on utilise 2.12
ThisBuild / scalaVersion := "2.12.21"

lazy val root = (project in file("."))
  .settings(
    name := "BUSI_Spark_Streming",


    // "%%" ajoute automatiquement la version de Scala au nom (spark-sql_2.12)
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql"  % "3.5.0" ,          // DataFrames + Structured Streaming (consumer)
      "org.apache.spark" %% "spark-core" % "3.5.0" ,          // RDD (producer)
      "com.microsoft.onnxruntime" % "onnxruntime" % "1.19.2", // execution du modele ONNX (consumer)
      "log4j" % "log4j" % "1.2.17",
      "com.typesafe" % "config" % "1.4.3"                     // lecture de application.conf
      ),

    // Spark 3.5 sur Java 17 : JVM separee avec les modules ouverts
    // (Java 17 interdit par defaut l'acces a certaines classes internes que Spark utilise)
    run / fork := true,
    run / javaOptions ++= Seq(
      "java.lang", "java.lang.invoke", "java.lang.reflect", "java.io", "java.net", "java.nio",
      "java.util", "java.util.concurrent", "java.util.concurrent.atomic",
      "sun.nio.ch", "sun.nio.cs", "sun.security.action", "sun.util.calendar"
    ).map(p => s"--add-opens=java.base/$p=ALL-UNNAMED") :+ "-Djdk.reflect.useDirectMethodHandle=false"
  )
