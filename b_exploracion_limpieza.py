###         LIBRERIAS
### --------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go 
import plotly.express as px
#from mlxtend.preprocessing import TransactionEncoder
import a_funciones as fn

###         CARGAR DATOS
### --------------------------------------------------------------------------------

# Crear cuando no existe el nombre de cd y para conectarse cuando sí existe.
conn=sql.connect('data\\db_movies') 
cur=conn.cursor() 
# Para funciones que ejecutan sql en base de datos

cur.execute('select name from sqlite_master where type = "table"')
cur.fetchall()

###         NOMBRAR LAS TABLAS
### --------------------------------------------------------------------------------

movies = pd.read_sql("SELECT * from movies", conn)
movies # Base de datos películas (MovieID, título, géneros)

ratings = pd.read_sql("SELECT * from ratings", conn)
ratings # Base de datos calificaciones (UserID, Movie ID, título, género)

###         TRATAMIENTO DE DATOS
### --------------------------------------------------------------------------------

## ----- TÍTULOS DUPLICADOS

dup_titles = pd.read_sql("""SELECT title, COUNT(*) as count
                         FROM movies
                         GROUP BY title
                         HAVING COUNT(*) > 1;""", conn)
dup_titles # Se identifican algunos títulos duplicados (5)

## ----- SEPARAR EL AÑO EN UNA NUEVA VARIABLE

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')  # Extrae el año
movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)  # Elimina el año del título

## ----- SEPARAR LOS GÉNEROS EN COLUMNAS

# Separar los géneros en columnas teniendo en cuenta el criterio de separación '|'
genres_dummies = movies['genres'].str.get_dummies(sep='|')

# Concatenar las columnas de géneros con el DataFrame original
movies2 = pd.concat([movies, genres_dummies], axis=1)

# Renombrar la variable de 'no genres listed'
movies2 = movies2.rename(columns={"(no genres listed)": "no_genre"})

# Eliminar la columna con los géneros unidos
movies2 = movies2.drop(columns=['genres'])

# Revisar las películas que no tienen género listado
no_genre = movies2[movies2['no_genre']==1]
no_genre

#vis = movies2.sort_values(by='year', ascending=True)
#print(movies[movies['movieId'] == 102747])

# Distribución de los géneros
genre_totals = movies2.iloc[:, 4:].sum().sort_values(ascending=False)
genre_totals_df = genre_totals.reset_index() #Convertir a dataframe
genre_totals_df.columns = ['Genre', 'Count']

## ----- SEPARAR LOS GÉNEROS EN COLUMNAS

# Crear el gráfico de barras
fig = go.Figure()

# Agregar datos al gráfico
fig.add_trace(go.Bar(
    x=genre_totals_df['Genre'],
    y=genre_totals_df['Count'],
    marker_color='springgreen'
))

# Actualizar el diseño del gráfico
fig.update_layout(
    title='Géneros de Películas',
    xaxis_title='Género',
    yaxis_title='Cantidad',
    xaxis_tickangle=-45,
    template='plotly_white'  # Plantilla del gráfico
)

# Mostrar el gráfico
fig.show()

###         VISUALIZACIÓN DE LOS DATOS
### --------------------------------------------------------------------------------

## ----- CALIFICACIONES GENERALES

cr = pd.read_sql("""
    SELECT 
        "rating" AS rating,
        COUNT(*) AS conteo,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS porcentaje
    FROM ratings
    GROUP BY "rating"
    ORDER BY "rating"
""", conn)
cr

pd.read_sql("select count(*) from ratings", conn)
# Se tiene un total de 100836 calificaciones

# Definir los colores según las calificaciones
colors = []
for rating in cr['rating']:
    if rating == 0.5:
        colors.append('#264653')  
    elif 1 <= rating <= 2:
        colors.append('#fe4a49')  
    elif 2.5 <= rating <= 3.5:
        colors.append('#fed766')  
    elif 4 <= rating <= 5:
        colors.append('#009fb7')  

data  = go.Bar( x=cr.rating,y=cr.conteo, text=cr.conteo, textposition="outside", marker_color=colors)

layout = go.Layout(
    title={
        'text': "Conteo de Calificaciones",
        'y': 0.94,
        'x': 0.5, 
        'xanchor': 'center',  # Anclar el título al centro
        'yanchor': 'top'
    },
    xaxis={
        'title': 'Calificación',
        'tickvals': cr['rating']  # Asegurar que todos los valores del eje X se muestren
    }, 
    yaxis={'title': 'Cantidad'},
    width=800,   # Ancho del gráfico
    height=600   # Alto del gráfico
)

# Crear la figura y mostrar
fig = go.Figure(data=data, layout=layout)
fig.show()

## ----- CALIFICACIONES DE LOS USUARIOS

rating_users=pd.read_sql(''' SELECT "userId" as user_id,
                         count(*) as cnt_rat
                         FROM ratings
                         group by "userId"
                         order by cnt_rat asc
                         ''',conn)

rating_users

# Histograma número de calificaciones por usuario 

fn.plot_histogram(rating_users, 'cnt_rat', bins=20, color='#264653')

# Filtro de cantidad de ratings

rating_users2=pd.read_sql(''' select "userId" as user_id,
                         count(*) as cnt_rat
                         FROM ratings
                         group by "userId"
                         having cnt_rat <=1000
                         order by cnt_rat asc
                         ''',conn )

rating_users2

# Histograma número de calificaciones por usuario 2

fn.plot_histogram(rating_users2, 'cnt_rat', bins=20, color='#264653')


#### verificar cuantas calificaciones tiene cada película

rating_movies=pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         order by cnt_rat desc
                         ''',conn )

rating_movies

# Histograma número de calificaciones por película

fn.plot_histogram(rating_movies, 'cnt_rat', bins=20, color='#264653')

# Filtro de cantidad de ratings por película

rating_movies2=pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         having cnt_rat >= 20
                         order by cnt_rat desc
                         ''',conn )

rating_movies2

# Histograma número de calificaciones por película 2

fn.plot_histogram(rating_movies2, 'cnt_rat', bins=20, color='#264653')


## ----- CALIFICACIONES DE LOS USUARIOS POR CADA CALIFICACIÓN

# Separar las calificaciones por puntaje
ratings_list = np.arange(0.5, 5.5, 0.5)

# Contar cuántas veces calificó cada usuario
rating_counts = ratings.groupby('userId')['rating'].value_counts().unstack(fill_value=0)

# Reindexar para asegurarse de que todas las calificaciones estén presentes
rating_counts = rating_counts.reindex(columns=ratings_list, fill_value=0)

# Añadir la columna de total de calificaciones
rating_counts['Total'] = rating_counts.sum(axis=1)

rating_counts = rating_counts.sort_values(by='Total', ascending=False)

rating_counts

# Crear una nueva matriz binaria
binary_matrix = (rating_counts.iloc[:, :-1] >= 1).astype(int)

# Calcular la columna Total en la matriz binaria
binary_matrix['Total'] = binary_matrix.sum(axis=1)

# Mostrar la nueva matriz con Total
binary_matrix

#Convertir matriz a dataframe
id_binary = binary_matrix.index.tolist()

binary_total = binary_matrix['Total'].tolist()

# Unión de ID con Total
df_binary = pd.DataFrame()
df_binary['userId'] = id_binary
df_binary['Total_ranges'] = binary_total
df_binary

# Pegar total de 'df_binary' en 'rating_counts' según 'userId'
df_merged_ratings = pd.merge(rating_counts,df_binary, on=['userId'], how='outer')
df_merged_ratings.sort_values(by='Total', ascending=False)

#Visualización cantidad de calificaciones por puntaje

# Contar cuántos usuarios tienen cada total
total_counts = binary_matrix['Total'].value_counts().sort_index()

# Graficar
fig = go.Figure()

fig.add_trace(go.Bar(
    x=total_counts.index,  # Total de calificaciones
    y=total_counts.values,  # Número de usuarios
    marker=dict(color='#264653')
))

# Personalizar el diseño del gráfico
fig.update_layout(
    title='Cantidad de Usuarios por Total de Calificaciones',
    xaxis_title='Total de Calificaciones',
    yaxis_title='Número de Usuarios',
    width=800,
    height=600,
)

# Mostrar el gráfico
fig.show()

# Visualización 2

# Definir los bins
bins = [0, 2, 4, 6, 11]  # Bins para los rangos (1-2), (3-4), (5-6), (+6)
labels = ['1-2', '3-4', '5-6', '+6']  # Etiquetas para los bins

# Agrupar los totales en los bins definidos
total_binned = pd.cut(total_counts.index, bins=bins, labels=labels, right=True)

# Contar cuántos usuarios caen en cada bin
binned_counts = total_counts.groupby(total_binned).sum().reindex(labels, fill_value=0)

# Graficar
fig = go.Figure()

fig.add_trace(go.Bar(
    x=binned_counts.index,  # Total de calificaciones en bins
    y=binned_counts.values,  # Número de usuarios en cada bin
    marker=dict(color='#264653')
))

# Personalizar el diseño del gráfico
fig.update_layout(
    title='Cantidad de Usuarios por Total de Calificaciones (Binned)',
    xaxis_title='Total de Calificaciones (bins)',
    yaxis_title='Número de Usuarios',
    width=800,
    height=600,
)

# Mostrar el gráfico
fig.show()

########################################################################################

###         P R E R P O C E S A M I E N T O
### --------------------------------------------------------------------------------

fn.ejecutar_sql('preprocesamiento.sql', cur)


cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()

### verficar tamaño de tablas con filtros ####

## ratings
pd.read_sql('select count(*) from ratings', conn)
pd.read_sql('select count(*) from ratings_filtered', conn)

## movies
pd.read_sql('select count(*) from movies', conn)
pd.read_sql('select count(*) from movies_sel', conn)


## tablas cruzadas ###
pd.read_sql('select count(*) from final_ratings', conn)

final=pd.read_sql('select * from final_ratings',conn)
final.duplicated().sum() ## al cruzar tablas a veces se duplican registros
final.info()
final.head(10)