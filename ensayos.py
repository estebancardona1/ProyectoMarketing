import ipywidgets as widgets
from IPython.display import display

# Inicializar la variable op
op = None

# Crear la lista desplegable
dropdown = widgets.Dropdown(
    options=['Amarillo', 'Azul', 'Rojo'],
    value='Amarillo',  # Valor inicial
    description='Colores:',
    disabled=False,
)

# Función para capturar el valor seleccionado
def on_change(change):
    global op  # Declarar op como variable global
    if change['type'] == 'change' and change['name'] == 'value':
        op = change['new']  # Almacenar la opción seleccionada en op
        print(f'Color seleccionado: {op}')

# Vincular la función a la lista desplegable
dropdown.observe(on_change)

# Mostrar la lista desplegable
display(dropdown)

if op == "Azul":
    print("gola")
else:
    print("nada")
    
    
    
##### INTERFAZ

import ipywidgets as widgets
from IPython.display import display

# Inicializar la variable op
op = None

# Crear la lista desplegable
dropdown = widgets.Dropdown(
    options=['Más Vistas', 'Mejores Calificadas', 
             'Año Específico', 'Género Específico'],
    value='Más Vistas',  # Valor inicial
    description='TOP 10:',
    disabled=False,
)

# Mostrar la lista desplegable
display(dropdown)



# Función para capturar el valor seleccionado
def on_change(change):
    global op  # Declarar op como variable global
    if change['type'] == 'change' and change['name'] == 'value':
        op = change['new']  # Almacenar la opción seleccionada en op
        if op == "Más Vistas":
        

# Vincular la función a la lista desplegable
dropdown.observe(on_change)

