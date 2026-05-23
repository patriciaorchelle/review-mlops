"""
Exemple Spark : traitement de données de reviews à grande échelle.

Ce script montre comment tu utiliserais Spark pour pré-traiter
des millions de reviews (là où Pandas serait trop lent).

Pour l'entretien : "J'ai utilisé Spark en cours et je suis capable
de faire du preprocessing distribué. Pour ce projet, Pandas était
suffisant (60 reviews), mais à l'échelle de Decathlon (millions
d'avis), Spark serait indispensable."

Note : PySpark doit être installé séparément :
    pip install pyspark
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline as SparkPipeline

def main():
    # ─── 1. CRÉER UNE SESSION SPARK ─────────────────────────────────────────
    # En local : spark utilise le CPU disponible
    # En prod : spark.master = "yarn" ou "k8s://..."
    spark = SparkSession.builder \
        .appName("Decathlon Review Preprocessing") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("Spark session créée")

    # ─── 2. CHARGER LES DONNÉES ─────────────────────────────────────────────
    # En prod : lire depuis GCS, S3, HDFS...
    # spark.read.parquet("gs://bucket/reviews/*.parquet")
    df = spark.read.csv("data/reviews.csv", header=True, inferSchema=True)

    print(f"Nombre de reviews : {df.count()}")
    df.show(5)

    # ─── 3. TRANSFORMATIONS DISTRIBUÉES ─────────────────────────────────────
    # Ce type de traitement peut tourner sur des milliards de lignes

    # Nettoyer le texte (minuscules, trim)
    df_clean = df \
        .withColumn("text_lower", F.lower(F.col("text"))) \
        .withColumn("text_clean", F.trim(F.col("text_lower"))) \
        .filter(F.length("text_clean") > 5)  # Supprimer les textes trop courts

    # Longueur des reviews
    df_features = df_clean \
        .withColumn("text_length", F.length("text_clean")) \
        .withColumn("word_count", F.size(F.split("text_clean", " ")))

    print("\nStatistiques par label :")
    df_features.groupBy("label").agg(
        F.count("*").alias("count"),
        F.avg("text_length").alias("avg_length"),
        F.avg("word_count").alias("avg_words")
    ).show()

    # ─── 4. PIPELINE ML SPARK (TF-IDF) ──────────────────────────────────────
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

    spark_pipeline = SparkPipeline(stages=[tokenizer, remover, hashingTF, idf])
    model = spark_pipeline.fit(df_features)
    df_vectorized = model.transform(df_features)

    print("\nFeatures TF-IDF créées :")
    df_vectorized.select("text", "label", "tfidf_features").show(3, truncate=True)

    # ─── 5. ÉCRIRE LES RÉSULTATS ────────────────────────────────────────────
    # En prod : écrire en Parquet pour la suite du pipeline
    # df_vectorized.write.parquet("data/processed/", mode="overwrite")

    spark.stop()
    print("\nTraitement Spark terminé !")


if __name__ == "__main__":
    main()
