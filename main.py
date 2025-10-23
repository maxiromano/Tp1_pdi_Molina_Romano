import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

"Función para equalización local de histograma, que recibre como parámetros la imagen a procesar y el tamaño de la ventana de procesaminto(tupla)"

def equalizador_local_histograma(imagen, tamaño_ventana):

    if len(imagen.shape) > 2: # validación
        raise ValueError("La imagen debe estar en escala de grises")

    M, N = tamaño_ventana # desempaqueta
    imagen_padding = cv2.copyMakeBorder(imagen, M//2, M//2, N//2, N//2, borderType=cv2.BORDER_REPLICATE) # padding (agrega pixeles a la imagen con el valor del cercano)
    ceros_imagen = np.zeros_like(imagen) # llenado con 0

    filas, columnas = imagen.shape # tamaño de la imagen para recorrer la misma

    for i in range(filas):
        for j in range(columnas):
            ventana_local = imagen_padding[i:i+M, j:j+N] # crop
            hist = cv2.calcHist([ventana_local], [0], None, [256], [0,256]) #aplicamos histograma
            hist = hist.ravel() / hist.sum() # normalización (vector plano hist / cantidad de elementos (MxN))
            cdf = hist.cumsum() # distribución acumulada(256 elementos, de 0 a 1)
            cdf_normalizada = (cdf * 255).astype('uint8') # mapea la probabilidad acumulada a la gama de intensidades de salida, formato uint8

            ceros_imagen[i, j] = cdf_normalizada[imagen[i, j]] # transformación final

    return ceros_imagen


img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

img_procesada = equalizador_local_histograma(img, (3, 3))

#local_hist_equalization(img, (8,8))
equalizador_local_histograma(img, (3,3))


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagen Procesada (Equalización Local)")
plt.imshow(img_procesada, cmap='gray')
plt.axis('off')

plt.show()

"""# Ejercicio 2

**DETECCIÓN DE LÍNEAS VERTICALES Y HORAZONTALES**
"""

# UMBRALADO BINARIO
def umbralado_binario(imagen,n_umbral):
  umbral = n_umbral
  img_bin = (imagen > umbral) * 255 # convierte true a 255 y false a 0
  img_bin  = img_bin.astype(np.uint8) # aseguramos formato uint8
  #plt.imshow(img_bin,cmap='gray',vmin=0,vmax=255)
  return img_bin

#DETECCIÓN DE LÍNEAS VERTICALES

def detectar_lineasv(imagen, porcentaje_minimov):

  lista_columnas = []
  alto, ancho = imagen.shape[:2]
  umbral_minimov = alto * porcentaje_minimov

  for j in range(ancho): # itera sobre cada columna
      conteo_pixeles_lineav = 0

      for i in range(alto):  # itera sobre cada fila dentro de esa columna
          if imagen[i][j] == 0:  # detecta si son todos los pixeles de la col 0.
              conteo_pixeles_lineav += 1

      if conteo_pixeles_lineav >= umbral_minimov:
        lista_columnas.append(j)

  return lista_columnas


# DETECCIÓN DE LÍNEAS HORIZONTALES
def detectar_lineash(imagen, porcentaje_minimoh):

  lista_filas = []
  alto, ancho = imagen.shape[:2]
  umbral_minimoh = alto * porcentaje_minimoh

  for i in range(alto): # itera sobre cada columna
      conteo_pixeles_lineah = 0

      for j in range(ancho):  # itera sobre cada fila dentro de esa columna
          if imagen[i][j] == 0:  # detecta si son todos los pixeles de la col 0.
              conteo_pixeles_lineah += 1

      if conteo_pixeles_lineah >= umbral_minimoh:
        lista_filas.append(i)

  return lista_filas

def agrupar_lineas_contiguas(lista_filas_detectadas):

    diccionario_lineas = {}
    grupo_actual = []
    id_linea = 0
    if not lista_filas_detectadas:
        return diccionario_lineas

    for fila_actual in lista_filas_detectadas:

        if not grupo_actual:
            grupo_actual.append(fila_actual)

        elif fila_actual == grupo_actual[-1] + 1:
            grupo_actual.append(fila_actual)

        else:
            inicio = grupo_actual[0]
            fin = grupo_actual[-1]
            diccionario_lineas[f"Linea_{id_linea}"] = (inicio, fin)
            id_linea += 1

            grupo_actual = [fila_actual]

    if grupo_actual: # último grupo después de salir del bucle
        inicio = grupo_actual[0]
        fin = grupo_actual[-1]
        diccionario_lineas[f"Linea_{id_linea}"] = (inicio, fin)

    return diccionario_lineas

img_formulario1 = cv2.imread('formulario_01.png',cv2.IMREAD_GRAYSCALE)
print(img_formulario1.shape) # tamaño de la imagen
print(np.unique(img_formulario1)) # valores únicos de intensidad
copia_img_form1 = img_formulario1.copy()
img_f1 = img_formulario1.flatten

img_formulario1_norm  = umbralado_binario(img_formulario1, 230)
lista_lineash = detectar_lineash(img_formulario1_norm, 0.3)
lista_lineasv = detectar_lineasv(img_formulario1_norm , 0.6)

lineash = agrupar_lineas_contiguas(lista_lineash)
lineasv = agrupar_lineas_contiguas(lista_lineasv)

print(lineasv)
print(lineash)

# CROP
# crop para cortar el encabezado "formulario a"
img_crop_form1 = img_formulario1_norm[lineash["Linea_1"][1]:lineash["Linea_13"][1],lineasv["Linea_1"][0]:lineasv["Linea_2"][0]]
img_crop_form1  = img_crop_form1.astype(np.uint8)

lista_lineashf = detectar_lineash(img_crop_form1, 0.6)

lineash_crop_final = agrupar_lineas_contiguas(lista_lineashf)

def deteccion_elementos(celda_img,    UMBRAL_ESPACIO_FIJO = 8):

    """
    Procesa una sub-imagen de celda de tabla para detectar caracteres, filtrar
    ruido/líneas y contar el número de caracteres y palabras usando un umbral
    de espacio adaptativo.
    """
    # Preprocesamiento e Identificación de Componentes
    # Inversión de la imagen para que las letras sean objetos claros

    celda_invertida = 255 - celda_img
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        celda_invertida, 8, cv2.CV_32S
    )

    # Excluir el fondo (índice 0)
    objetos_stats = stats[1:]

    if len(objetos_stats) == 0:
        return (f"Cantidad total de caracteres: 0",
                f"Cantidad de espacios 'en blanco' (separaciones entre palabras): 0",
                f"Cantidad de palabras formadas: 0")

    MIN_AREA_CH = 50
    MAX_AREA_CH = 500
    MIN_altura_CH = 10
    MAX_ancho_CH = 100

    componentes_validas = []

    for s in objetos_stats:
        x, y, w, h, area = s[0], s[1], s[2], s[3], s[4]

        if (area >= MIN_AREA_CH and area <= MAX_AREA_CH and
            h >= MIN_altura_CH and w <= MAX_ancho_CH):
            componentes_validas.append({'x': x, 'x_fin': x + w})

    componentes_ordenadas = sorted(componentes_validas, key=lambda c: c['x'])
    cantidad_caracteres = len(componentes_ordenadas)

    if cantidad_caracteres <= 1:
        return (cantidad_caracteres, cantidad_caracteres)

    espacios_en_blanco = 0
    palabras_formadas = 1


    for i in range(cantidad_caracteres - 1):
        comp_actual = componentes_ordenadas[i]
        comp_siguiente = componentes_ordenadas[i+1]

        distancia = comp_siguiente['x'] - comp_actual['x_fin']

        if distancia >= UMBRAL_ESPACIO_FIJO:
            espacios_en_blanco += 1
            palabras_formadas += 1

    return(cantidad_caracteres, palabras_formadas)


# Funciones de validacion

def validar_nombre_apellido(caracteres, palabras):
    return "OK" if (palabras >= 2 and caracteres <= 25) else "MAL"

def validar_edad(caracteres, palabras):
    return "OK" if (2 <= caracteres <= 3 and palabras == 1) else "MAL"

def validar_mail(caracteres, palabras):
    return "OK" if (palabras == 1 and caracteres <= 25) else "MAL"

def validar_legajo(caracteres, palabras):
    return "OK" if (caracteres == 7 and palabras == 1) else "MAL"

def validar_comentarios(caracteres, palabras):
    return "OK" if (palabras >= 1 and caracteres <= 25) else "MAL"

def validar_pregunta(caracteres, palabras):
    return "OK" if (palabras == 1 and caracteres == 1) else "MAL"

def procesar_formulario(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    # Umbralado
    img_bin = umbralado_binario(img, 230)

    # Detectar líneas y agrupar
    lineash_original = agrupar_lineas_contiguas(detectar_lineash(img_bin, 0.3))
    lineasv_original = agrupar_lineas_contiguas(detectar_lineasv(img_bin, 0.6))


    ultima_linea_y_pos = img_bin.shape[0]
    primer_linea = lineash_original["Linea_1"][1]

    # crop para sacar encabezados
    img_crop_form1 = img_bin[primer_linea:ultima_linea_y_pos,
                             lineasv_original["Linea_1"][0]:lineasv_original["Linea_2"][0]]
    img_crop_form1 = img_crop_form1.astype(np.uint8)

    # Recortes
    nombre_y_apellido_crop = img_crop_form1[0 : lineash_crop_final["Linea_1"][0], :]

    edad = img_crop_form1[lineash_crop_final["Linea_1"][1] : lineash_crop_final["Linea_2"][0], :]
    mail = img_crop_form1[lineash_crop_final["Linea_2"][1] : lineash_crop_final["Linea_3"][0], :]
    legajo = img_crop_form1[lineash_crop_final["Linea_3"][1] : lineash_crop_final["Linea_4"][0], :]
    preg1 = img_crop_form1[lineash_crop_final["Linea_5"][1] : lineash_crop_final["Linea_6"][0], :]
    preg2 = img_crop_form1[lineash_crop_final["Linea_6"][1] : lineash_crop_final["Linea_7"][0], :]
    preg3 = img_crop_form1[lineash_crop_final["Linea_7"][1] : lineash_crop_final["Linea_8"][0], :]
    comentarios = img_crop_form1[lineash_crop_final["Linea_8"][1] : lineash_crop_final["Linea_9"][0], :]

    # detección y validacion
    nombre_info = deteccion_elementos(nombre_y_apellido_crop)
    edad_info = deteccion_elementos(edad)
    mail_info = deteccion_elementos(mail, UMBRAL_ESPACIO_FIJO=30)
    legajo_info = deteccion_elementos(legajo, UMBRAL_ESPACIO_FIJO=30)
    preg1_info = deteccion_elementos(preg1)
    preg2_info = deteccion_elementos(preg2)
    preg3_info = deteccion_elementos(preg3)
    comentarios_info = deteccion_elementos(comentarios)


    resultados_valid = {
        "Nombre y Apellido": validar_nombre_apellido(*nombre_info),
        "Edad": validar_edad(*edad_info),
        "Mail": validar_mail(*mail_info),
        "Legajo": validar_legajo(*legajo_info),
        "Pregunta 1": validar_pregunta(*preg1_info),
        "Pregunta 2": validar_pregunta(*preg2_info),
        "Pregunta 3": validar_pregunta(*preg3_info),
        "Comentarios": validar_comentarios(*comentarios_info),
    }

    # Determinar si el formulario es globalmente correcto
    formulario_es_correcto = all(estado == "OK" for estado in resultados_valid.values())
    estado_global = "OK" if formulario_es_correcto else "MAL"

    print(f"\nResultados de validación para {ruta}: ")
    for campo, estado in resultados_valid.items():
        print(f"> {campo}: {estado}")
    print(f"Estado global del formulario: {estado_global}")

    return nombre_y_apellido_crop, estado_global, resultados_valid

resultados_imagen = []
resultados_para_csv = []

for i in range(1, 6):
    ruta_imagen = f"formulario_{i:02}.png"
    id_formulario = i
    recorte_nombre_apellido, estado_formulario, detalles_resultados = procesar_formulario(ruta_imagen)
    resultados_imagen.append({
        'recorte': recorte_nombre_apellido,
        'estado': estado_formulario,
        'nombre_formulario': ruta_imagen
    })

    # csv
    fila_csv = {'ID': id_formulario}
    columnas_ordenadas = ["Nombre y Apellido", "Edad", "Mail", "Legajo",
                         "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"]
    for columna in columnas_ordenadas:
        fila_csv[columna] = detalles_resultados.get(columna)
    resultados_para_csv.append(fila_csv)

# creamos imagen resumen
ancho_maximo = 0
alto_maximo = 0
for resultado in resultados_imagen:
    alto, ancho = resultado['recorte'].shape
    ancho_maximo = max(ancho_maximo, ancho)
    alto_maximo = max(alto_maximo, alto)

altura_slot = alto_maximo + 30  # espacio ok/mal
ancho_slot = ancho_maximo + 50

altura_imagen_total = len(resultados_imagen) * altura_slot
ancho_imagen_total = ancho_slot + 150

# creamos imagen en blanco
imagen_resumen = np.full((altura_imagen_total, ancho_imagen_total, 3), 255, dtype=np.uint8)

desplazamiento_y = 0
for resultado in resultados_imagen:
    imagen_recorte = resultado['recorte']
    estado = resultado['estado']

    recorte_color = cv2.cvtColor(imagen_recorte, cv2.COLOR_GRAY2BGR)

    alto_recorte_actual, ancho_recorte_actual = recorte_color.shape[:2]

    inicio_y_pegado = desplazamiento_y + (alto_maximo - alto_recorte_actual) // 2
    imagen_resumen[inicio_y_pegado:inicio_y_pegado + alto_recorte_actual, 20:20 + ancho_recorte_actual] = recorte_color

    color_estado = (0, 255, 0) if estado == "OK" else (0, 0, 255)  # verde OK, rojo MAL
    texto_estado = f"Formulario {resultado['nombre_formulario'].split('_')[1].split('.')[0]}: {estado}"

    cv2.putText(imagen_resumen, texto_estado,
                (ancho_maximo + 40, desplazamiento_y + alto_maximo // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2, cv2.LINE_AA)

    desplazamiento_y += altura_slot

# Mostrar y guardar la imagen de resumen
plt.figure(figsize=(10, altura_imagen_total / 100))
plt.imshow(cv2.cvtColor(imagen_resumen, cv2.COLOR_BGR2RGB))
plt.title("Resumen de Validación de Formularios")
plt.axis("off")
plt.show()
cv2.imwrite("resumen_validacion_formularios.png", imagen_resumen)

# Guardar resultados en CSV
df_resultados = pd.DataFrame(resultados_para_csv)

orden_columnas_final = ['ID', "Nombre y Apellido", "Edad", "Mail", "Legajo",
                        "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"]
df_resultados = df_resultados[orden_columnas_final]

nombre_archivo_csv = "resultados_validacion_formularios.csv"
df_resultados.to_csv(nombre_archivo_csv, index=False)


