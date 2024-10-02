import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib


#### conectar_base_de_Datos

conn=sql.connect('data\\df_movies')
cur=conn.cursor()

#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()