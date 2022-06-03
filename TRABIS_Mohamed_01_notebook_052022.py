#!/usr/bin/env python
# coding: utf-8

# Chargement des packages

# Fonctions pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, length, col
from pyspark.sql.types import ArrayType, StringType, IntegerType, DoubleType, DataType, FloatType
from pyspark.ml.image import ImageSchema
from pyspark.ml.feature import PCA
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit

from pyspark import SparkContext

# Fonction pour ouvrir l'image à partir de son chemin d'accès
from PIL import Image

# Librairies classiques
import numpy as np
import sys
import os
import io
import time
from io import StringIO
import pyarrow as pa
import pyarrow.parquet as pq

# Librairie pour se connecter au service S3 d'AWS
import boto3

from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions


# Fonction de chargement des données et extraction de features avec vgg16

data_folder = "data2/Test/Avocado/"
bucket = 'trabisfruit'

def vgg_extract():
    # modele pour extraire les features, derniere couche.
    conv_base = VGG16(
        include_top=False,
        weights=None,
        pooling='max',
        input_shape=(100, 100, 3))


# Ouvre l'image via la librairie pillow et resize l'image pour des raisons de mémoires

    s3 = boto3.resource("s3", region_name='eu-west-1')
    bucket = s3.Bucket("trabisfruit")
    prefix = data_folder

    list_path_img = []
    for file in bucket.objects.filter(Prefix=prefix).limit(10):

        if (file.key != prefix):
            obj = bucket.Object(file.key)
            label = file.key.split('/')[-2]
            response = obj.get()
            file_stream = response['Body']
            img = Image.open(file_stream)
            #img = img.resize((20, 20))
            # convert image to flatten array
            flat_array = np.array(img).ravel().tolist()
            tensor = np.array(flat_array).reshape(1, 100, 100, 3).astype(np.uint8)
            # preprocess input
            prep_tensor = preprocess_input(tensor)
            features = conv_base.predict(prep_tensor).ravel().tolist()
            #print(len(features))
            # Store file key and features
            list_path_img.append((file.key, label, features))
    print("Nombre d'images chargees :", len(list_path_img))


    # Create spark dataframe from previous list of tuples
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    spark = SparkSession.builder.appName("name").getOrCreate()
    df = spark.createDataFrame(list_path_img, ['path_img', 'label', 'vgg_features'])

    return df



def pca_transformation(df, n_components=2, col_image='vgg_features'):
    """
    Applique un algorithme de PCA sur l'ensemble des images pour réduire la dimension de chaque image
    du jeu de données.

    Paramètres:
    df(pyspark dataFrame): contient une colonne avec les données images
    n_components(int): nombre de dimensions à conserver
    col_image(string): nom de la colonne où récupérer les données images
    """

    # Initilisation du temps de calcul
    start_time = time.time()

    # Les données images sont converties au format vecteur dense
    #ud_f = udf(lambda l: Vectors.dense(l), VectorUDT())
    #df = df.withColumn('image_pca', ud_f(col_image))

    # from Array to Vectors for PCA
    array_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df = df.withColumn('vgg_vectors', array_to_vector_udf(col_image))

    #df = df.select('path_img', 'label', 'vgg_vectors')


    # reduce with PCA - k=20
    #pca = PCA(k=512, inputCol='vgg_vectors').fit(df)
    #print(pca.explainedVariance)


    pca = PCA(k=n_components, inputCol='vgg_vectors', outputCol='pca_vectors')
    model = pca.fit(df)
    df = model.transform(df)

    # from Vector to Array
    vector_to_array_udf = udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    df = df.withColumn('pca_vectors', vector_to_array_udf('pca_vectors'))

    #df = df.select('path_img', 'label', 'pca_vectors')

    # Affiche le temps de calcul
    print("Temps d'execution PCA {:.2f} secondes".format(time.time() - start_time))

    return df


def write_pandas_csv_to_s3(df, bucketName, keyName, fileName):
    # Télécharger le ficher csv en local
    df.toPandas().to_csv('df2.csv')

    # transférer le fichier dans s3
    s3 = boto3.client("s3")
    BucketName = bucketName
    with open(fileName) as f:
       object_data = f.read()
       s3.put_object(Body=object_data, Bucket=BucketName, Key=keyName)

    os.remove(fileName)


if __name__ == "__main__":

    # Définis le chemin d'accès au dossier des images
    # Chemins différents suivant si le script est executé en local ou sur AWS
    try:
        if sys.argv[1] == 'True':
            folder = data_folder
        else:
            folder = "s3://trabisfruit/"
    except:
        sys.exit(0)
    print(folder)

    # Démarre la session Spark
    try:
        sc = SparkContext.getOrCreate()
        sc.setLogLevel('WARN')
        spark = SparkSession.builder.appName("name").getOrCreate()
    except:
        print("Erreur à la construction du moteur spark")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("----Chargement des images et extraction des features avec VGG16-----")
    # Initilisation du temps de calcul pour l'enregistrement
    start_time = time.time()

    df = vgg_extract()
    df.show()
    df.printSchema()
    # Affiche le temps de calcul de l'écriture des résultats
    print("Temps d'execution d'extraction des features avec VGG16 : {:.2f} secondes".format(time.time() - start_time))


    print("---Réduction dimmensionnelle---")
    df = pca_transformation(df)
    df.show()
    df.printSchema()


    print("---- Enregistrement des résultats dans S3 ----")
    # Initilisation du temps de calcul pour l'enregistrement
    start_time = time.time()

    # enregistrement des données dans S3 (format csv)
    df = df.select('pca_vectors')
    #write_pandas_csv_to_s3(df, "trabisfruit", "outputs/df2.csv", "df2.csv")

    # Affiche le temps de calcul de l'écriture des résultats
    print("Temps d'execution de l'enregistrement des résultats dans S3 : {:.2f} secondes".format(time.time() - start_time))