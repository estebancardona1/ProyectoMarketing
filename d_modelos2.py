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

###         CARGAR DATOS
### --------------------------------------------------------------------------------

conn=sql.connect('data\\db_movies')
cur=conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

####################################################################################
########### 3. SISTEMAS DE RECOMENDACIÓN BASADO EN CONTENIDO KNN ###################
############### CON BASE EN EL CONTENIDO VISTO POR EL USUARIO ######################
####################################################################################

## ----- CARGAR LOS DATOS OBTENIDOS EN CONSULTAS PREVIAS

##### cargar data frame escalado y con dummies ###
movies_dum= joblib.load('salidas\\movies_dum2.joblib')

### carga data frame normal que tiene nombres de libros
movies_final=joblib.load('salidas\\movies_final.joblib')

## ----- SELECCIONAR USUARIOS PARA LA RECOMENDACIÓN

usuarios=pd.read_sql('select distinct (user_id) as user_id from final_ratings',conn)

def recomendar(user_id=sorted(list(usuarios['user_id'].value_counts().index))):
    
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select *from final_ratings where user_id=:user',conn, params={'user':user_id,})
    
    ###convertir ratings del usuario a array
    l_movies_r=ratings['movie_id'].to_numpy()
    
    ###agregar la columna de movieId y titulo de la película a dummie para filtrar y mostrar nombre
    movies_dum[['movieId','title']]=movies_final[['movieId','title']]
    
    ### filtrar películas calificados por el usuario
    movies_r=movies_dum[movies_dum['movieId'].isin(l_movies_r)]
    
    ## eliminar columna title e movieId
    movies_r=movies_r.drop(columns=['movieId','title'])
    movies_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    
    ##centroide o perfil del usuario
    centroide=movies_r.groupby("indice").mean()
    
    ### filtrar pelícuas no vistas
    movies_nr=movies_dum[~movies_dum['movieId'].isin(l_movies_r)]
    
    ## eliminar nombre e movieId
    movies_nr=movies_nr.drop(columns=['movieId','title'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nr)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend=movies_final.loc[ids][['title','year']]
    
    # Obtener el título de la primera película recomendada
    top_movie = recomend.iloc[0]['title']
    print("Basado en tus gustos creemos que te podría gustar: ", top_movie)
    fn.post_img(top_movie)  # Mostrar imagen del póster de la película recomendada
    print("Además algunos títulos adicionales que te podrían gustar: ")
    
    
    vistos=movies_final[movies_final['movieId'].isin(l_movies_r)][['title','movieId']]
    
    return recomend


print(interact(recomendar))

# SANTI ------------------------------------------------------------------------------------------

##### cargar data frame escalado y con dummies ###
movies_dum= joblib.load('salidas\\movies_dum2.joblib')

df_complete= joblib.load('salidas\\df_complete.joblib')
df_complete.info()
df_complete['year']=df_complete.year.astype('int')

#### seleccionar usuario para recomendaciones ####
usuarios=pd.read_sql('select distinct (user_id) as user_id from final_ratings',conn)

user_id = 400

def recomendar(user_id=list(usuarios['user_id'].value_counts().index)):
    
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select *from final_ratings where user_id=:user',conn, params={'user':user_id,})
    
    ###convertir ratings del usuario a array
    l_movies_r=ratings['movie_id'].to_numpy()
    
    ###agregar la columna de isbn y titulo del libro a dummie para filtrar y mostrar nombre
    movies_dum[['movie_id','title']]=df_complete[['movie_id','title']]
    
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
    recomend_movies=df_complete.loc[ids][['title']]
    vistos=df_complete[df_complete['movie_id'].isin(l_movies_r)][['title']]
    
    # Obtener el título de la primera película recomendada
    top_movie = recomend_movies.iloc[0]['title']
    print("Basado en tus gustos creemos que te podría gustar: ", top_movie)
    fn.post_img(top_movie)  # Mostrar imagen del póster de la película recomendada
    print("Además algunos títulos adicionales que te podrían gustar: ")
    
    return recomend_movies

print(interact(recomendar))

#---------------------------------------------------------------------------------------------------

####################################################################################
############## 4. SISTEMA DE RECOMENDACIÓN FILTRO COLABORATIVO #####################
####################################################################################

## ----- SELECCIONAR LA BASE DE DATOS CON LAS CALIFICACIONES
ratings = pd.read_sql('SELECT * FROM final_ratings', conn)

####los datos deben ser leidos en un formato espacial para surprise
reader = Reader(rating_scale=(1, 5)) ### la escala de la calificación
###las columnas deben estar en orden estándar: user item rating
data   = Dataset.load_from_df(ratings[['user_id','movie_id','movie_rating']], reader)

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
def recomendaciones(user_id=list(usuarios['user_id'].unique()), n_recomend=(1, 20)):
    
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
    
    # Extraer el año del título y eliminarlo del nombre
    recomendados['year'] = recomendados['title'].str.extract(r'\((\d{4})\)')  # Extrae el año
    recomendados['title'] = recomendados['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)  # Elimina el año del título
    
    # Renombrar la columna 'est' a 'score sup'
    recomendados = recomendados.rename(columns={'est': 'score sup'})
    
    # Eliminar las columnas 'iid' y 'index'
    recomendados = recomendados.drop(columns=['iid', 'index'], errors='ignore')

    # Obtener el título de la primera película recomendada
    top_movie = recomendados.iloc[0]['title']
    print("Basado en tus gustos creemos que te podría gustar: ", top_movie)
    fn.post_img(top_movie)  # Mostrar imagen del póster de la película recomendada
    print("Además algunos títulos adicionales que te podrían gustar: ")
    
    # Devolver el DataFrame de recomendaciones sin mostrar el index en la salida
    return recomendados.reset_index(drop=True)


# Interactuar con la función
print(interact(recomendaciones))

 
 