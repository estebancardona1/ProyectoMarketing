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