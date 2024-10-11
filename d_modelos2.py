import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib
import a_funciones as fn
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import train_test_split
import re

###         CARGAR DATOS
### --------------------------------------------------------------------------------

conn=sql.connect('data\\db_movies')
cur=conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()


# ----- FUNCIÓN PARA CORREGIR ALGUNOS TÍTULOS

def fix_titles(title):
    # Usamos una expresión regular para identificar títulos como "Anaconda, The" o "Matrix, The"
    pattern = r'(.+),\s(The|A|An)$'
    match = re.match(pattern, title)
    
    # Si encontramos una coincidencia, reordenamos el título
    if match:
        new_title = f"{match.group(2)} {match.group(1)}"
        return new_title
    else:
        return title




####################################################################################
########### 3. SISTEMAS DE RECOMENDACIÓN BASADO EN CONTENIDO KNN ###################
############### CON BASE EN EL CONTENIDO VISTO POR EL USUARIO ######################
####################################################################################

## ----- CARGAR LOS DATOS OBTENIDOS EN CONSULTAS PREVIAS

##### cargar data frame escalado y con dummies ###
movies_dum= joblib.load('salidas\\movies_dum.joblib')

### carga data frame normal que tiene nombres de películas
movies_final=joblib.load('salidas\\movies_only.joblib')

#ratings=pd.read_sql('select * from full_ratings', conn)

# EJEMPLO RELEVANTE !!!
#ratings[ratings['userId']==604]

## ----- SELECCIONAR USUARIOS PARA LA RECOMENDACIÓN

usuarios=pd.read_sql('select distinct userId from full_ratings',conn)

userId = 1

def recomendar(userId=sorted(list(usuarios['userId'].value_counts().index))):
    
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select * from full_ratings where userId=:user',conn, params={'user':userId,})
    
    ###convertir ratings del usuario a array
    l_movies_w=ratings['movieId'].to_numpy()
    
    ###agregar la columna de movieId y titulo de la película a dummie para filtrar y mostrar nombre
    movies_dum[['movieId','title']]=movies_final[['movieId','title']]
    
    ### filtrar películas calificados por el usuario
    movies_w=movies_dum[movies_dum['movieId'].isin(l_movies_w)]
    
    ## eliminar columna title e movieId
    movies_w=movies_w.drop(columns=['movieId','title'])
    movies_w["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    
    ##centroide o perfil del usuario
    centroide=movies_w.groupby("indice").mean()
    
    ### filtrar pelícuas no vistas
    movies_nw=movies_dum[~movies_dum['movieId'].isin(l_movies_w)]
    
    ## eliminar nombre e movieId
    movies_nw=movies_nw.drop(columns=['movieId','title'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nw)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend=movies_final.loc[ids][['movieId','title','year']]
    
    ## ENSAYOOOOO
    rating_data = pd.read_sql('''SELECT movieId,ROUND(AVG(rating), 2) as score, 
                                COUNT(rating) as views FROM full_ratings
                                GROUP BY movieId''', conn)
    
    
    ### Unir los datos de rating y views a las recomendaciones
    recomend = recomend.merge(rating_data, on='movieId', how='left')
    
    recomend = recomend.drop(columns=['movieId'])
    
    recomend = recomend[recomend['score'] >= 3]
    
    recomend = recomend[recomend['views'] >= 20]
    
    ## ENSAYOOOOO
    
    # Obtener el título de la primera película recomendada
    top_movie = recomend.iloc[0]['title']
    print("Basado en tus gustos creemos que te podría gustar: ", top_movie)
    fn.post_img(top_movie)  # Mostrar imagen del póster de la película recomendada
    print("Además algunos títulos adicionales que te podrían gustar: ")
    
    
    vistos=movies_final[movies_final['movieId'].isin(l_movies_w)][['title','movieId']]
    
    return recomend


print(interact(recomendar))

#---------------------------------------------------------------------------------------------------

####################################################################################
############## 4. SISTEMA DE RECOMENDACIÓN FILTRO COLABORATIVO #####################
####################################################################################

## ----- SELECCIONAR LA BASE DE DATOS CON LAS CALIFICACIONES
ratings = pd.read_sql('SELECT * FROM ratings_final', conn)

####los datos deben ser leidos en un formato espacial para surprise
reader = Reader(rating_scale=(1, 5)) ### la escala de la calificación
###las columnas deben estar en orden estándar: user item rating
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)


#####Existen varios modelos 
models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline()]
results = {}

###knnBasiscs: calcula el rating ponderando por distancia con usuario/Items
###KnnWith means: en la ponderación se resta la media del rating, y al final se suma la media general
####KnnwithZscores: estandariza el rating restando media y dividiendo por desviación 
####Knnbaseline: calculan el desvío de cada calificación con respecto al promedio y con base en esos calculan la ponderación

#### for para probar varios modelos ##########
model=models[1]
for model in models:
 
    CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)  
    
    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
             rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    results[str(model).split("algorithms.")[1].split("object ")[0]] = result


performance_df = pd.DataFrame.from_dict(results).T
performance_df.sort_values(by='RMSE')

###################se escoge el mejor knn Baseline#########################
param_grid = { 'sim_options' : {'name': ['msd','cosine'], \
                                'min_support': [5,2], \
                                'user_based': [False, True]}
             }

## min support es la cantidad de items o usuarios que necesita para calcular recomendación
## name medidas de distancia

### se afina si es basado en usuario o basado en ítem

gridsearchKNNBaseline = GridSearchCV(KNNBaseline, param_grid, measures=['rmse'], \
                                      cv=2, n_jobs=-1)
                                    
gridsearchKNNBaseline.fit(data)


gridsearchKNNBaseline.best_params["rmse"]
gridsearchKNNBaseline.best_score["rmse"]
gs_model=gridsearchKNNBaseline.best_estimator['rmse'] ### mejor estimador de gridsearch

################# Entrenar con todos los datos y Realizar predicciones con el modelo afinado

trainset = data.build_full_trainset() ### esta función convierte todos los datos en entrnamiento, las funciones anteriores dividen  en entrenamiento y evaluación
model=gs_model.fit(trainset) ## se reentrena sobre todos los datos posibles (sin dividir)


joblib.dump(model, "salidas\\recom_model.joblib")


predset = trainset.build_anti_testset() ### crea una tabla con todos los usuarios y los libros que no han leido
#### en la columna de rating pone el promedio de todos los rating, en caso de que no pueda calcularlo para un item-usuario
len(predset)

predictions = gs_model.test(predset) ### función muy pesada, hace las predicciones de rating para todos los libros que no hay leido un usuario
### la funcion test recibe un test set constriuido con build_test method, o el que genera crosvalidate


predictions[0:10] 
####### la predicción se puede hacer para un libro puntual
model.predict(uid=1, iid='1',r_ui='') ### uid debía estar en número e isb en comillas

predictions_df = pd.DataFrame(predictions) ### esta tabla se puede llevar a una base donde estarán todas las predicciones
predictions_df.shape
predictions_df.head()
predictions_df['r_ui'].unique() ### promedio de ratings
predictions_df.sort_values(by='est',ascending=False)


##### funcion para recomendar los 10 libros con mejores predicciones y llevar base de datos para consultar resto de información
def recomendaciones(user_id=list(usuarios['userId'].unique()), n_recomend=(1, 20)):
    
    # Filtrar las predicciones para el usuario seleccionado y obtener las mejores n recomendaciones
    predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                    sort_values(by="est", ascending = False).head(n_recomend)

    recomendados = predictions_userID[['iid', 'est']]
    recomendados.to_sql('reco', conn, if_exists="replace")
    
    # Unir con la tabla de películas para obtener los títulos
    recomendados = pd.read_sql('''SELECT a.*, b.title 
                                  FROM reco a 
                                  LEFT JOIN movies_final b
                                  ON a.iid = b.movieId''', conn)
    
    # Renombrar la columna 'est' a 'score sup'
    recomendados = recomendados.rename(columns={'est': 'score sup'})
    
    # Eliminar las columnas 'iid' y 'index'
    recomendados = recomendados.drop(columns=['iid', 'index'], errors='ignore')
    
    recomendados['title'] = recomendados['title'].apply(fix_titles)

    # Obtener el título de la primera película recomendada
    top_movie = recomendados.iloc[0]['title']
    print("Basado en tus gustos creemos que te podría gustar: ", top_movie)
    fn.post_img(top_movie)  # Mostrar imagen del póster de la película recomendada
    print("Además algunos títulos adicionales que te podrían gustar: ")
    
    # Devolver el DataFrame de recomendaciones sin mostrar el index en la salida
    return recomendados.reset_index(drop=True)


# Interactuar con la función
print(interact(recomendaciones))

