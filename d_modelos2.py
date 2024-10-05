import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib
####Paquete para sistemas de recomendación surprise
###Puede generar problemas en instalación local de pyhton. Genera error instalando con pip
#### probar que les funcione para la próxima clase 

from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import train_test_split

#############################################
#### conectar_base_de_Datos#################
############################################

conn=sql.connect('data\\db_movies')
cur=conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

#######################################################################
#### 3 Sistema de recomendación basado en contenido KNN #################
#### Con base en todo lo visto por el usuario #######################
#######################################################################

##### cargar data frame escalado y con dummies ###
movies_dum= joblib.load('salidas\\movies_dum2.joblib')

df_final3= joblib.load('salidas\\df_final3.joblib')
df_final3.info()
df_final3['year']=df_final3.year.astype('int')

#### seleccionar usuario para recomendaciones ####
usuarios=pd.read_sql('select distinct (user_id) as user_id from final_ratings',conn)

user_id = 400

def recomendar(user_id=list(usuarios['user_id'].value_counts().index)):
    
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select *from final_ratings where user_id=:user',conn, params={'user':user_id,})
    
    ###convertir ratings del usuario a array
    l_movies_r=ratings['movie_id'].to_numpy()
    
    ###agregar la columna de isbn y titulo del libro a dummie para filtrar y mostrar nombre
    movies_dum[['movie_id','title']]=df_final3[['movie_id','title']]
    
    ### filtrar libros calificados por el usuario
    movies_r=movies_dum[movies_dum['movie_id'].isin(l_movies_r)]
    
    ## eliminar columna nombre e isbn
    movies_r=movies_r.drop(columns=['movie_id','title'])
    movies_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    ##centroide o perfil del usuario
    centroide=movies_r.groupby("indice").mean()
    
    
    ### filtrar libros no leídos
    movies_nv=movies_dum[~movies_dum['movie_id'].isin(l_movies_r)]
    ## eliminbar nombre e id
    movies_nv=movies_nv.drop(columns=['movie_id','title'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nv)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend_movies=df_final3.loc[ids][['title']]
    vistos=df_final3[df_final3['movie_id'].isin(l_movies_r)][['title']]
    
    return recomend_movies

print(interact(recomendar))

