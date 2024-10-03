def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)
    
def plot_histogram(dataframe, count_column, bins=20, color='#264653'):
    
    import plotly.graph_objs as go

    # Graficar la distribución
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=dataframe[count_column],
        nbinsx=bins,
        marker=dict(color=color, line=dict(color='black', width=1)),
        opacity=0.75
    ))

    # Personalizar el diseño del gráfico
    fig.update_layout(
        title='Histograma de número de calificaciones por usuario',
        xaxis_title='Número de calificaciones',
        yaxis_title='Frecuencia',
        bargap=0.2,
        width=800,   
        height=600   
    )

    # Mostrar el gráfico
    fig.show()
    
#BUSCAR IMAGEN

def fetch_movie_poster(movie_title):
    
    import requests
    from bs4 import BeautifulSoup
    from IPython.display import display, HTML
    
    search_title = movie_title.replace(" ", "+")  # Reemplazar espacios por + para la URL

    # URL de búsqueda de TMDb
    search_url = f"https://www.themoviedb.org/search?query={search_title}"

    # Hacer la solicitud
    response = requests.get(search_url)

    # Verificar que la solicitud fue exitosa
    if response.status_code == 200:
        # Analizar el contenido HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Encontrar el primer resultado de la búsqueda
        movie_card = soup.find('div', class_='card')

        if movie_card:
            # Obtener el enlace de la película
            movie_link = movie_card.find('a')['href']
            full_movie_url = f"https://www.themoviedb.org{movie_link}"

            # Hacer una solicitud a la página de la película
            movie_response = requests.get(full_movie_url)

            if movie_response.status_code == 200:
                movie_soup = BeautifulSoup(movie_response.text, 'html.parser')

                # Encontrar el póster
                poster = movie_soup.find('img', class_='poster')
                if poster:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster['src']}"
                    
                    # Mostrar la imagen en un tamaño más pequeño usando HTML
                    display(HTML(f'<img src="{poster_url}" width="150" style="margin: 5px;" />'))
                else:
                    print(f"No se encontró un póster para {movie_title}.")
            else:
                print(f"Error al acceder a la página de la película: {movie_response.status_code}")
        else:
            print(f"No se encontraron resultados para la búsqueda de {movie_title}.")
    else:
        print(f"Error en la búsqueda de {movie_title}: {response.status_code}")
        
        
