# Manejo de redes neuronales con Keras
<details>

<summary>Índice de contenidos</summary>

* [***Datasets: train(ing), validation & test***](#datasets-training-validation--test)
  * [***Training dataset***](#training-dataset)
  * [***Validation dataset***](#validation-dataset)
  * [***Test dataset***](#test-dataset)

* [***Resolviendo un problema de clasificacion binaria***](#resolviendo-un-problema-de-clasificacion-binaria)  
  * [***Obtención de datos IMDB - Keras***](#obtención-de-datos-imdb---keras)  
  * [***Normalizado de datos***](#normalizado-de-datos)
    * [***Variables cualitativas***](#variables-cualitativas)
    * [***Variables cuantitativas***](#variables-cuantitativas)
  * [***Arquitectura del modelo***](#arquitectura-del-modelo)

* [***Entrenamiento del modelo de clasificación binaria***](#entrenamiento-del-modelo-de-clasificación-binaria)
  * [***Entrenando el modelo***](#entrenando-el-modelo)  
  * [***Analizando resultados***](#analizando-resultados)

* [***Regularización - Dropout***](#regularización---dropout)  
  * [***Regularización L2 (Ridge)***](#regularización-l2-ridge)  
  * [***Regularización L1 (Lasso)***](#regularización-l1-lasso)
  * [***Regularización ElasticNet (L1 y L2)***](#regularización-elasticnet-l1-y-l2)
  * [***Dropout***](#dropout)
  * [***Normalización por lotes (Batch normalization)***](#normalización-por-lotes-batch-normalization)
  * [***Data augmentation***](#data-augmentation)
  * [***Early Stopping***](#early-stopping)

* [***Reduciendo el overfitting***](#reduciendo-el-overfitting)  

* [***Resolviendo un problema de clasificación múltiple***](#resolviendo-un-problema-de-clasificación-múltiple)  

* [***Entrenamiento del modelo de clasificación múltiple***](#entrenamiento-del-modelo-de-clasificación-múltiple)  

* [***Validación de nuestro modelo usando Cross Validation***](#validación-de-nuestro-modelo-usando-cross-validation)
  * [***Tipos de validación***](#tipos-de-validación)
    * [***Hold-Out***](#hold-out)
    * [***K-Folds***](#k-folds)
    * [***LOOCV***](#loocv)

* [***Resolviendo un problema de regresión***](#resolviendo-un-problema-de-regresión)

* [***Entrenamiento del modelo de regresión***](#entrenamiento-del-modelo-de-regresión)

* [***Análisis de resultados del modelo de regresión***](#análisis-de-resultados-del-modelo-de-regresión) 

</details>

## Datasets: train(ing), validation & test  

### Training dataset

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Training dataset</span>][r000]***: El ***modelo se ajusta inicialmente a un conjunto de datos de entrenamiento***​, que es un conjunto de ejemplos utilizados para ***ajustar los parámetros (pesos y sesgos) del modelo***. El modelo se ejecuta con este conjunto de datos y produce un resultado, que luego se compara con el objetivo, para cada vector de entrada del conjunto de datos. En función del resultado de la comparación y del algoritmo de aprendizaje específico utilizado, se ajustan los parámetros del modelo. El ajuste del modelo puede incluir tanto la selección de variables como la estimación de parámetros.

El modelo aprende de este dataset.

### Validation dataset

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Validation dataset</span>][r001]***: El modelo de entrenamiento ajustado se utiliza para predecir las respuestas de las observaciones en un segundo conjunto de datos denominado conjunto de datos de validación. Este proporciona una ***evaluación no sesgada del ajuste de un modelo en el conjunto de datos de entrenamiento*** mientras ***se ajustan los hiperparámetros del modelo​*** (por ejemplo, el número de unidades ocultas -capas y anchos de capa- en una ***RNA​***). Estos conjuntos de datos pueden utilizarse para la regularización mediante la detención temprana (detener el entrenamiento cuando aumenta el error en el conjunto de datos de validación, ya que es un signo de ***ajuste excesivo*** al conjunto de datos de entrenamiento). Este sencillo procedimiento se complica en la práctica por el hecho de que el error de este conjunto de datos puede fluctuar durante el entrenamiento, produciendo múltiples mínimos locales. Esta complicación ha llevado a la creación de muchas reglas ad hoc para decidir cuándo ha comenzado realmente el ***sobreajuste***.

Es decir, los conjuntos de datos de validación son utilizados para realizar una ***evaluación no sesgada de un modelo entrenado***, mientras son ajustados los hiperparámetros. La evaluación se vuelve más sesgada a medida que la habilidad en el dataset de validación se incorpora a la configuración del modelo.

El modelo nunca "aprende" del dataset de validación. Utilizamos los resultados de este dataset y ajustamos (actualizamos) hiperparámetros. 

Afecta a un modelo, pero solo indirectamente. El conjunto de validación también se conoce como ***dataset de desarrollo***, ya que se utiliza durante la etapa de "desarrollo" del modelo.
 
### Test dataset

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Test dataset</span>][r002]***: Es un conjunto de datos utilizado para proporcionar una ***evaluación imparcial del ajuste final de un modelo en el conjunto de datos de entrenamiento***.​ Si los datos del conjunto de datos de prueba nunca se han utilizado en el entrenamiento (por ejemplo, en la ***validación cruzada***), el conjunto de datos de prueba también se denomina ***conjunto de datos retenidos***. El término "conjunto de validación" se utiliza a veces en lugar de "conjunto de prueba" en algunas publicaciones (por ejemplo, si el conjunto de datos original se dividió en sólo dos subconjuntos, el conjunto de prueba podría denominarse conjunto de validación).

Proporcionan el ***estándar utilizado para evaluar el modelo***. Se usa una vez que un modelo está completamente entrenado (entrenamiento y validación). 

Generalmente se utiliza para evaluar modelos que compiten. Por ejemplo, en muchas competiciones de Kaggle, el dataset de validación se lanza inicialmente, junto con el dataset de entrenamiento. El dataset de pruebas (real), únicamente se lanza cuando la competención está a punto de cerrar, y es este el que decide el ganador.  

***Muchas veces el dataset de validación se usa como dataset de pruebas, pero no es una buena práctica***. ***Debe contener datos cuidadosamente muestreados***, que abarquen las diversas clases que enfrentará el modelo, cuando se utilice en el mundo real.

![Test dataset][i000]  

## Resolviendo un problema de clasificacion binaria

Para resolver un problema de clasificacion binaria vamos a utilizar el dataset [IMDB movie review sentiment classification](https://keras.io/api/datasets/imdb/), el cual contiene una muestra de 25.000 reseñas de películas de IMDB etiquetadas por sentimiento (positivo/negativo). Adicionalmente, contiene 10.000 de las palabras más usadas en cada una de las reseñas. Estas palabras están guardadas en un catálogo que asigna un índice a cada una de las palabras más empleadas.

El objetivo de esta práctica es utilizar nuestros conocimientos de normalización de datos, para preparar este dataset para ser clasificado por una ***RNA*** que pueda predecir el sentimiento de una nueva reseña.  

### Obtención de datos IMDB - Keras 

```python
# Carga del dataset
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Aspecto de uno de los datos de entrenamiento
print("Train data example")
print(train_data[0])
print("Train label example")
print(train_labels[0])
```

Respuesta:
```commandline
Train data example
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
Train label example
1
```

Sin embargo, nos gustaría ver, con fines didácticos, cuál es el verdadero texto que tiene una muestra del dataset de entrenamiento. Para ello nos apoyaremos de la función `get_word_index()`, la cual, nos retornará el catalogo de las palabras utilizadas con su respectivo índice, después convertiremos este `word_index` en un diccionario. Así, acceederemos, de forma sencilla a las palabras del `word_index` de acuerdo al índice de la palabra.  

```python
def convert_number_to_word(example):
    word_index = imdb.get_word_index()
    word_index = dict([(value, key) for (key, value) in word_index.items()])
    print(" ".join([str(word_index.get(_ - 3)) for _ in example]))

# Aspecto real de un texto de entrada
convert_number_to_word(train_data[0])
```

Respuesta:
```commandline
None this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert None is an amazing actor and now the same being director None father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for None and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also None to the two little boy's that played the None of norman and paul they were just brilliant children are often left out of the None list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all
```

### Normalización de datos

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Normalización de datos</span>][r003]***: El objetivo de la normalización es ***cambiar los valores de las columnas numéricas del conjunto de datos para usar una escala común, sin distorsionar las diferencias en los intervalos de valores ni perder información***. La normalización también es necesaria para que algunos algoritmos modelen los datos correctamente.

Por ejemplo, supongamos que el conjunto de datos de entrada contiene una columna con valores comprendidos entre 0 y 1, y otra columna con valores comprendidos entre 10.000 y 100.000. La enorme diferencia en el escala de los números podría generar problemas al intentar combinar los valores como características durante el modelado.

La normalización evita estos problemas mediante la ***creación de nuevos valores que mantienen la distribución general y las relaciones en los datos de origen***, a la vez que se ***conservan los valores dentro de una escala*** que ***se aplica en todas las columnas numéricas que se usan en el modelo***.

En nuestro ejemplo, las palabras del dataset han sido guardadas como índices en lugar de utilizar las palabras directamente, porque, en general, en cualquier problema de ***ML/DL***, tenemos que los datos que vamos a utilizar para entrenar a nuestro modelo de IA pueden ser:

- ***Estructurados***: Datos que tengan forma de fila - columna (similar a una tabla de excel)
- ***No estructurados***: Datos que explícitamente NO tengan una forma de fila - columna, cómo lo son: El audio, las imágenes o el texto.

A su vez, los datasets ***estructurados*** cuentan con dos tipos principales de variables: **Cualitativas y Cuantitativas**.

![Tipos de variables][i001]  

#### Variables cualitativas

Este tipo de variables representan características o cualidades de un objeto y no pueden ser medidas de forma numerica.

Tipos:
- ***Ordinal***: A pesar de NO tener un valor asignado, sí posee un orden conocido (por ejemplo, muy frio, frio, templado, caliente, muy caliente).
- ***Nominal***: No poseen un valor asignado y tampoco cuentan con un orden conocido (por ejemplo, rojo, azul, verde).

#### Variables cuantitativas

Representan cantidades numéricas, lo cual les permite utilizar operaciones aritméticas.  

Tipos:
- ***Discreta***: Se puede contar con números enteros (por ejemplo, años de vida, número de hijos en una familia, etc.)
- ***Continúa***: Se debe expresar con números decimales (por ejemplo, el peso de una persona puede ser 64,512 Kg o 72,018 Kg etc.)

Teniendo en cuenta lo anterior, podemos observar que la propia naturaleza de las ***RNA***, incentiva el uso de variables cuantitativas continuas para ser utilizadas por el modelo. Sin embargo, ya sabemos que el texto es un tipo de dato `No estructurado` y esto NO le gusta a los modelos ***DL***, entonces tendríamos, primero que usar algún mecanismo que nos permita estructurar este tipo de dato y de forma preferible llevarlo a una variable cuantitativa continua. 

El texto puede ser tratado como una variable ***cualitativa nominal***. Entonces es fácil ver que una forma de convertir variables ***nominales*** a ***discretas*** es creando un catálogo de índices, que logre mapear cada valor nominal a un número discreto. ***[gato, azul, coche]*** se puede convertir en ***[1, 2, 3]*** dado un diccionario de índices-palabras como el siguiente:

`{1: "gato", 2: "azul", 3: "coche"}`

A este tipo de conversión entre variables nominales y discretas se le conoce como: ***label encoding***. [Acceder aquí para saber más de label encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

Sin embargo, pese a que esta técnica es muy fácil de implementar y es bastante eficiente en terminos de tiempo y memoria, tiene una gran desventaja y es que este tipo de transformaciones podría hacernos pensar que un ***coche*** es equivalente a tres ***gatos*** y numéricamente es cierto, pero sabemos que esa analogía no tiene ningún sentido. 

Una alternativa al ***Label Encoding*** es el método llamado ***One Hot Encoding***. En esta estrategia se crea una columna para cada valor distinto que exista en la característica que estamos codificando y, para cada registro, se marca con un 1 la columna a la que pertenezca dicho registro y se deja las demás con 0. En el ejemplo visto en la sección anterior en el que codificábamos el sexo de los pasajeros, ***One Hot Encoding*** generaría dos columnas (una para el valor "male" y otra para el valor "female") y, para cada pasajero, asignaría un 1 a la columna de "male" y un 0 a la de "female" en el caso de ser un hombre, y viceversa en el caso de ser mujer.

En el ejemplo anterior las variables nominales de: `[gato, azul, coche]` en lugar de ser representadas utilizando ***Label enconding*** como `{1: "gato", 2: "azul", 3: "coche"}` podrían tener la siguiente reprsentación de ***One Hot Encoding***

```
gato =  [1, 0, 0]
azul =  [0, 1, 0]
coche = [0, 0, 1]
```

Esta es una mejor aproximación, puesto que entonces un coche tiene el mismo peso que un gato. Y todas las palabras tienen la misma distancia entre ellas. Sin embargo, esta NO es la representación definitiva para la conversión de texto a un tipo de dato estructurado. Un ejemplo sencillo es que en el lenguaje existe un concepto llamado semántica que nos permite darnos cuenta que:

Entre las palabras ***cuchillo*** y ***tenedor*** hay menos distancia que entre ***cuchillo*** y ***ornitorrico***. Esta explicación es la introducción al tema de word embeddings.  

Ahora que ya tenemos un conocimiento sobre qué es ***Label Encoding*** y ***One Hot Encoding*** podemos proceder a transformar la representación de ***Label Encoding*** que utilizo el dataset de IMDB por una versión en ***One Hot Encoding***, la cuál es una mejor representación, puesto que tiene la forma de un ***tensor*** que es ideal para ***DL***.  

```python

def one_hot_encoding(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1
    return results

# Covertirmos la representacion de label encoding a one hot encoding para nuestros parámetros de entrada
x_train = one_hot_encoding(train_data)
x_test = one_hot_encoding(test_data)

# Siempre que podamos transformemos los datos enteros en float32 puesto que tensorflow esta optimizado para trabajar con este tipo de datos
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(train_data.shape)
print(x_train.shape)
print(x_train[0], x_train[0].shape)
```

Resultado:
```commandline
(25000,)
(25000, 10000)
[0. 1. 1. ... 0. 0. 0.] (10000,)
```

### Arquitectura del modelo

Para esta arquitectura tenemos un modelo de 2 capas ocultas, cada una con 16 neuronas y con ***función de activación*** ***ReLU***. Nuestra capa de clasificación solo tendrá 1 neurona, puesto que es suficiente para un problema de clasificación binaria y tendrá una ***función de activación*** ***Sigmoide***.  

Finalmente, el modelo será compilado con el optimizador `rmsprop` y como ***función de perdida*** usaremos `binary_crossentropy`, la cual funciona muy bien para problemas de clasificación binaria, y como medida de desempeño usaremos `accuracy`

```python
def architecture(model: models.Sequential, input_shape) -> models.Sequential:
    model.add(layers.Dense(16, activation="relu", input_shape=input_shape))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


model = models.Sequential()
model = architecture(model, (10000, ))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
```

## Entrenamiento del modelo de clasificación binaria  

### Entrenando el modelo

Entrenaremos el modelo durante 50 épocas, con un batch size de 512 y finalmente, usaremos un 30% de los datos de entrenamiento, como datos de validación. Esto nos permitirá observar si el modelo tiene o no ***overfitting***.

```python
history = model.fit(x_train, y_train, epochs=50, batch_size=512, validation_split=0.3)
```

### Analizando resultados


```python
results = model.evaluate(x_test, y_test)
print(results)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

fig = plt.figure(figsize=(10, 10))
epoch = range(1, len(loss_values) + 1)

plt.plot(epoch, loss_values, 'o-r', label='training')
plt.plot(epoch, val_loss_values, '--', label='validation')
plt.title("Error in training and validation datasets")
plt.xlabel("epochs")
plt.ylabel("Binary Cross Entropy")
plt.legend()
```

Resultado:
```commandline
782/782 [==============================] - 1s 1ms/step - loss: 1.2614 - acc: 0.8542
```

![Errores][i002]  

## Regularización - Dropout

El ***overfitting*** (sobre ajuste), sucede cuando el modelo no es capaz de generalizar el conocimiento adquirido en la etapa de ***trainig***, es decir, no aprende (recuerda).  

Este problema se observa, cuando comparamos las ***funciones de coste*** a lo largo de las épocas, entre el ***trainig dataset*** y el ***validation dataset***.  

En nuestro problema de clasificación binaria (de reseñas de IMDB), en el gráfico anterior, la curva roja representa el error en el ***trainig dataset*** y la azul el error en el ***validation dataset*** y la azul. A lo largo de las primeras 10 épocas observamos como, poco a poco, el error va disminuyendo en el ***trainig dataset***, pero sucede lo contrario en el ***validation dataset***. Aproximadamente, después de la época 4, el error en el ***validation dataset*** empieza a subir. Es decir, detectamos que hay ***overfitting***.

En ***DL*** contamos con una serie de técnicas, para reducir el ***overfitting***, las describimos a continuación.

### Regularización L2 (Ridge)

Reduce el valor de los parámetros para que sean pequeños.

Esta técnica introduce un término adicional de penalización en la función de ***coste original*** (L), añadiendo a su valor la ***suma de los cuadrados*** de los parámetros (ω).

Pero este nuevo término puede ser alto; tanto que la red minimizaría la ***función de coste*** haciendo los parámetros muy cercanos a 0, lo que no sería nada conveniente. Es por ello que multiplicaremos ese sumando por una constante (λ) pequeña, cuyo valor escogeremos de forma arbitraria (0.1, 0.01, …).

La función de coste queda, por tanto, así:

![L2](https://latex2png.com/pngs/3457ebd84be9eb2d9e89ed7df3349a77.png)

**¿Cuándo es efectiva Ridge (L2)?**

Ridge nos va a servir de ayuda cuando sospechemos que varios de los atributos de entrada (features) estén correlacionados entre ellos. Ridge hace que los coeficientes acaben siendo más pequeños. Esta disminución de los coeficientes minimiza el efecto de la correlación entre los atributos de entrada y hace que el modelo generalice mejor. Ridge funciona mejor cuando la mayoría de los atributos son relevantes.

### Regularización L1 (Lasso)

Existe otra técnica muy parecida a la anterior denominada regularización L1 donde los parámetros en el sumatorio del término de penalización no se elevan al cuadrado, sino que se usa su valor absoluto.

![L1](https://latex2png.com/pngs/fb0f2fb18c9fc8d7db7288c0fb99fad3.png)

Esta variante empuja el valor de los parámetros hacia valores más pequeños, haciendo incluso que la influencia de algunas variables de entrada sea nula en la salida de la red, lo que supone una selección de variables automática. El resultado es una mejor generalización, pero solo hasta cierto punto (la elección del valor de λ cobra más importancia en este caso).

**¿Cuándo es efectiva Lasso (L1)?**

Lasso nos va a servir de ayuda cuando sospechemos que varios de los atributos de entrada (features) sean irrelevantes. Al usar Lasso, estamos fomentando que la solución sea poco densa. Es decir, favorecemos que algunos de los coeficientes acaben valiendo 0. Esto puede ser útil para descubrir cuáles de los atributos de entrada son relevantes y, en general, para obtener un modelo que generalice mejor. Lasso nos puede ayudar, en este sentido, a hacer la selección de atributos de entrada. Lasso funciona mejor cuando los atributos no están muy correlacionados entre ellos.

### Regularización ElasticNet (L1 y L2)

ElasticNet combina las regularizaciones L1 y L2. Con el parámetro r podemos indicar que importancia relativa tienen Lasso y Ridge respectivamente. Matemáticamente:

![elastic](https://latex2png.com/pngs/ec1b4f7f8e2e71c50ae9957e4c41afe4.png)

**¿Cuándo es efectiva ElasticNet?**

Usaremos ElasticNet cuando tengamos un gran número de atributos. Algunos de ellos serán irrelevantes y otros estarán correlacionados entre ellos.

### Dropout

Esta técnica difiere de las vistas hasta el momento. El procedimiento es sencillo: por cada nueva entrada a la red en fase de entrenamiento, se desactivará aleatoriamente un porcentaje de las neuronas en cada capa oculta, acorde a una probabilidad de descarte previamente definida. Dicha probabilidad puede ser igual para toda la red, o distinta en cada capa.

![dropout.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fdropout.png)

Lo que se consigue con esto es que ninguna neurona memorice parte de la entrada; que es precisamente lo que sucede cuando tenemos sobreajuste.

Una vez tengamos el modelo listo para realizar predicciones sobre muestras nuevas, debemos compensar de alguna manera el hecho de que no todas las neuronas permanecieran activas en entrenamiento, ya que en inferencia sí que estarán todas funcionando y, por tanto, habrá más activaciones contribuyendo a la salida de la red. Un ejemplo de dicha compensación podría ser multiplicar todos los parámetros por la probabilidad de no descarte.

### Normalización por lotes (Batch normalization)

La historia de esta técnica es curiosa. Se presentó como una solución para reducir algo llamado Internal Covariate Shift, pero parece que no es eso lo que hace. Aún así es una técnica esencial para redes neuronales por todo lo que aporta, como explicamos a continuación.

La normalización en lotes consiste básicamente en añadir un paso extra, habitualmente entre las neuronas y la función de activación, con la idea de normalizar las activaciones de salida. Lo ideal es que la normalización se hiciera usando la media y la varianza de todo el conjunto de entrenamiento, pero si estamos aplicando el descenso del gradiente estocástico para entrenar la red, se usará la media y la varianza de cada mini-lote de entrada.

Nota: cada salida de cada neurona se normalizará de forma independiente, lo que quiere decir que en cada iteración se calculará la media y la varianza de cada salida para el mini-lote en curso.

A continuación de la normalización se añaden 2 parámetros: un bias como sumando, y otra constante similar a un bias, pero que aparece multiplicando cada activación. Esto se hace para que el rango de la entrada escale fácilmente hasta el rango de salida, lo que ayudará mucho a nuestra red a la hora de ajustar a los datos de entrada, y reducirá las oscilaciones de la función de coste. Como consecuencia de esto podremos aumentar la tasa de aprendizaje (no hay tanto riesgo de acabar en un mínimo local) y la convergencia hacia el mínimo global se producirá más rápidamente.

![batch_norm.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fbatch_norm.png)

La normalización por lotes es más una técnica de ayuda al entrenamiento que una estrategia de regularización en sí misma. Esto último se logra realmente aplicando algo adicional conocido como momentum. La idea de este momentum es que cuando introduzcamos un nuevo mini-batch de entrada (N muestras procesadas en paralelo) no se usen una media y una desviación muy distintas a las de la iteración anterior, para lo que se tendrá en cuenta el histórico, y se elegirá una constante que pondere la importancia de los valores del mini-batch actual frente a los valores del anterior. Gracias a todo esto se conseguirá reducir el sobreajuste.

### Data augmentation

La idea es aplicar diversas transformaciones sobre las entradas originales, obteniendo muestras ligeramente diferentes, pero iguales en esencia, lo que permite a la red desenvolverse mejor en la fase de inferencia.

Esta técnica se utiliza mucho en el campo de la visión artificial porque funciona de maravilla (en otros campos está por explorar). Dentro de dicho contexto, una misma imagen de entrada será procesada por la red neuronal tantas veces como epochs ejecutemos en entrenamiento; provocando que la red acabe memorizando la imagen si estamos entrenamos demasiado. Lo que haremos es aplicar transformaciones de forma aleatoria cada vez que volvamos a introducir la imagen a la red.

Ejemplos de transformaciones son:

- Voltear la imagen en horizontal / vertical.
- Rotar la imagen X grados.
- Recortar, añadir relleno, redimensionar.
- Aplicar deformaciones de perspectiva.
- Ajustar brillo, contraste, saturación.
- Introducir ruido, defectos.
- Combinaciones de las anteriores.

![data_augmentation.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fdata_augmentation.png)

De esta forma contaremos con más información para entrenamiento sin necesidad de obtener muestras adicionales, y también sin alargar los tiempos. Lo mejor es que si por ejemplo nuestra red se dedica a clasificar imágenes o detectar objetos, esta técnica conseguirá que el modelo sea capaz de obtener buenos resultados para imágenes tomadas desde distintos ángulos o bajo distintas condiciones de luz. Por tanto, conseguiremos que la red no sobreajuste y que generalice mejor.

### Early Stopping 

Es una técnica que trata de aplicar ciertas reglas para saber cuándo es momento de parar el entrenamiento, de forma que no se produzca sobreajuste a los datos de entrada, ni tampoco subajuste.

La regla más extendida sería la de entrenar el modelo monitorizando su rendimiento y guardando sus parámetros al finalizar cada epoch, hasta que apreciemos que el error de validación aumenta de forma sostenida (hay empeoramientos que son debidos a la componente estocástica del algoritmo). Nos quedaremos con el modelo que teníamos justo en el momento anterior.

## Reduciendo el overfitting

De forma completamente didactica y para poder observar como se utilizan algunas de las técnicas de normalización anteriormente mencionadas, vamos a crear un nuevo modelo con una nueva arquitectura que tenga las siguientes características:

- Un modelo más simple: Menos cantidad de neuronas por capa.
- Uso de regularizadores l1 y l2
- Uso de BatchNormalization
- Uso de dropout
- Early Stopping - Entrenar con menos épocas. 

Primero definimos la nueva arquitectura del modelo:

```python
def architecture(model: models.Sequential) -> models.Sequential:
    model.add(layers.Dense(8, activation="relu", input_shape=(10000,), kernel_regularizer=regularizers.l1_l2()))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation="relu", kernel_regularizer=regularizers.l1_l2()))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
```

Y ahora procedemos a compilar

```python
model_norm = models.Sequential()
model_norm = architecture(model_norm)
model_norm.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history_norm = model_norm.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.3)
```

Finalmente evaluamos los resultados del nuevo modelo:

```python
results = model_norm.evaluate(x_test, y_test)
print(results)
history_dict = history_norm.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

fig2 = plt.figure(figsize=(10, 10))
epoch = range(1, len(loss_values) + 1)
plt.plot(epoch, loss_values, 'o-r', label='training')
plt.plot(epoch, val_loss_values, '--', label='validation')
plt.title("Error in training and validation datasets using normalization")
plt.xlabel("epochs")
plt.ylabel("Binary Cross Entropy")
plt.legend()
plt.savefig("imgs/errores_norm.png")
plt.close()
```

Respuseta:
```commandline
782/782 [==============================] - 1s 1ms/step - loss: 0.4057 - acc: 0.8629
```
![errores_norm.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fclasificacion%20binaria%2Fimgs%2Ferrores_norm.png)

El modelo ahora ha obtenido una performance ligeramente superior en el dataset de testing pasando de 0.85 a 0.86 con un modelo más simple, y en la gráfica de comparación del error de training y validation vemos una disminución de overfitting.

## Resolviendo un problema de clasificación múltiple

Este problema en esencia es MUY similar al problema anterior, simplemente vamos a explicar en qué consiste este nuevo problema de clasificación y solamente voy a poner los puntos diferenciadores respecto al caso de estudio anterior.

En esta ocasión el problema consiste en entrenar a un modelo de deep learning que sea capaz de diferenciar entre 46 tópicos diferentes relacionados con temas de noticias. Para ello vamos a utilizar el dataset [Reuters](https://keras.io/api/datasets/reuters/).

Este dataset cuenta con propiedades similares al dataset anterior, también está representado como ***label encoding*** y por ende utilizaremos las mismas técnicas de preprocesamiento de datos para transformarlo en ***one hot encoding***. Sin embargo, el hecho de que la clasificación sea multiple, genera un par de diferencias importantes respecto a la clasificación binaria. Veamos los pasos a continuación.

1. Importando bibliotecas:

```python
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import reuters
from keras import layers, models
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import regularizers
```
2. Cargando Dataset:

```python
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
```

3. Normalizando datos:

```python
def one_hot_encoding(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1
    return results


x_train = one_hot_encoding(train_data)
x_test = one_hot_encoding(test_data)

# Aquí hay un cambio interesante respecto al ejemplo anterior:
# Observemos como luce una etiqueta de y_train

print(train_labels[0], train_labels[0].shape)

# Debemos transformar esta salida en una salida de clasificación multiple, esto es lo mismo
# que hicimos en el problema de clasificación de números escritos a mano del dataset MNIST
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print(y_train[0], y_train[0].shape)
```

Resultados:
```commandline
3 ()
[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] (46,)
```

4. Definiendo la Arquitectura de nuestra red

```python
def architecture(model: models.Sequential, input_shape: tuple, n_classes: int) -> models.Sequential:
    model.add(layers.Dense(128, activation="relu", input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    # IMPORTANTE: Ahora que nuestro problema es de clasificación MULTIPLE nuestra activación de la capa de predicción
    # Es diferente, en este caso usamos softmax, porque nos interesa tener la probabilidad de cada clase a la salida.
    model.add(layers.Dense(n_classes, activation="softmax"))
    return model


model_norm = models.Sequential()
model_norm = architecture(model=model_norm, input_shape=(10000, ), n_classes=46)
```

## Entrenamiento del modelo de clasificación múltiple

5. Compilando la red:

```python
# Dado que nuestro problema tiene varias clases, entonces usaremos "categorical_crossentropy"
# en lugar de "binary_crossentropy"
model_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
```

6. Entrenando la red:

```python
history_norm = model_norm.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.3)
```

7. Análisis de resultados:

```python
results = model_norm.evaluate(x_test, y_test)
print(results)
history_dict = history_norm.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict["acc"]
val_acc_values = history_dict["val_acc"]
epoch = range(1, len(loss_values) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
fig.suptitle("Neural Network's Result")
ax1.set_title("Loss function over epoch")
ax2.set_title("Acc over epoch")
ax1.set(ylabel="loss", xlabel="epochs")
ax2.set(ylabel="acc", xlabel="epochs")
ax1.plot(epoch, loss_values, 'o-r', label='training')
ax1.plot(epoch, val_loss_values, '--', label='validation')
ax2.plot(epoch, acc_values, 'o-r', label='training')
ax2.plot(epoch, val_acc_values, '--', label='validation')
ax1.legend()
ax2.legend()
plt.savefig("imgs/results.png")
plt.close()
```

Resultados:
```commandline
71/71 [==============================] - 0s 1ms/step - loss: 1.2337 - acc: 0.8045
```
![results.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fclasificaci%C3%B3n%20multiple%2Fimgs%2Fresults.png)

## Validación de nuestro modelo usando Cross Validation

Antes de continuar con este tema, es un buen momento para repasar conceptos de machine learning. En este momento hablaremos un poco sobre la estrategia de ***Cross Validation*** la cual implementaremos en este mini proyecto. 


- **La última palabra siempre la van a tener los datos.**
  - Todas nuestras intuiciones no tiene nada que hacer frente a lo que digan los datos y las matemáticas que aplicamos sobre estos datos. Por eso es importante siempre tener rigurosidad a la hora de evaluar los resultados que estamos recibiendo.

- **Necesitamos mentalidad de testeo.**
  - No se trata solamente de probar un poco al principio y un poco al final, sino que tendremos que probar constantemente durante todo el proceso, para poder encontrar cuál es la solución óptima que realmente nos soluciona el problema que tenemos pendiente, todo esto:
    - con varias formas
    - con varios conjuntos de datos
    - con varias configuraciones de parámetros
    - con varias distribuciones de nuestros datos

- **Todos los modelos son malos, solamente algunos son útiles.**
  - Todos los modelos que nosotros hacemos en últimas son una sobre simplificación de lo que pasa realmente. Entonces nunca nuestros modelos van a corresponder con la realidad al cien por ciento. Si jugamos lo suficiente y si somos lo suficientemente hábiles para configurar, vamos a llegar a un punto donde el modelo que estamos trabajando va a ser útil para ciertos casos específicos dentro del mundo real.

### Tipos de validación

#### Hold-Out

Se trata de dividir nuestros datos entrenamiento/pruebas, básicamente consiste en usar porcentajes fijos, por lo regular 70% de entrenamiento y 30% de pruebas.

![ho1.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fho1.png)

![ho2.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fho2.png)

**¿Cuándo utilizar Hold-out?**

- Se requiere un prototipado rápido.
- No se tiene mucho conocimiento en ML.
- No se cuenta con abundante poder de cómputo.

#### K-Folds

Usar validación cursada K-Fold, aquí vamos a plegar nuestros datos k veces, el k es un parámetro que nosotros definimos y en esos pliegues vamos a utilizar diferentes partes de nuestro dataset como entrenamiento y como test, de tal manera que intentemos cubrir todos los datos de entrenamiento y de test, al finalizar el proceso.

![kf1.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fkf1.png)

![kf2.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fkf2.png)

**¿Cuándo utilizar K-Folds?**

- Recomendable en la mayoría de los casos.
- Se cuenta con un equipo suficiente para desarrollar ML.
- Se require la integración con técnicas de optimización paramétrica.
- Se tiene más tiempo para las pruebas.

#### LOOCV

Validación cruzada LOOCV, Leave One Out Cross Validation. Este es el método más intensivo, ya que haremos una partición entre entrenamiento y pruebas, porque vamos a hacer entrenamiento con todos los datos, salvo 1 y vamos a repetir este proceso tantas veces hasta que todos los datos hayan sido probados.

![lo.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Flo.png)

**¿Cuándo utilizar LOOCV?**

- Se tiene gran poder de cómputo
- Se cuenta con pocos datos para poder dividir por Train/Test
- Cuando se quiere probar todos los casos posibles (para personas con TOC)

## Resolviendo un problema de regresión

Tomando lo aprendido en la sección anterior, en esta ocasión vamos a resolver un problema de ***house pricing***, para ello vamos a utilizar el dataset de [Boston Housing price regression dataset](https://keras.io/api/datasets/boston_housing/) el cual contiene 404 muestras de casas, cada casa contiene 13 variables continuas que expresan atributos de la casa.

Nuestra tarea entonces será predecir el valor de la casa con base en los 13 atributos mencionados anteriormente. La metodología como es costumbre será similar a los ejemplos anteriores, sin embargo; para este específico mini proyecto utilizaremos una estrategia de validación del modelo diferente. 

En este ejemplo usaremos ***k-Fold Cross Validation*** descrito en la clase anterior. Para ello crearemos una función que contenga el modelo, su arquitectura y compilado y lo correremos para cada uno de los `folds` de validación.

***Nota***: El código completo lo puedes encontrar [aquí](3%20Manejo%20de%20redes%20neuronales%20con%20Keras/regresión/main.py)

1. **Importando bibliotecas necesarias:**
    ```python
    import pandas as pd
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import numpy as np
    from keras.datasets import boston_housing
    from keras import models, layers
    # Biblioteca para implementar K-Fold CrossValidation
    from sklearn.model_selection import KFold
    # Biblioteca para la correcta normalización de datos numéricos
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    ```
    Lo más relevante a resaltar aquí es el uso de KFold y StandardScaler provenientes de `sklearn`


2. **Definiendo la arquitectura del modelo:**

    La siguiente función se encarga de crear la arquitectura y compilar al nuestro modelo de regresión:
    ```python
    def build_model_regression(dim):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_dim=dim))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        # Como la última capa es una predicción de regresión, NO necesita una capa de activación
        model.add(layers.Dense(1))
        # El error sí será el mean squared error, pero la métrica debe ser diferente, en este caso max absolute error
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model
    ```
    Los puntos claves a destacar son: 
    - La función de perdida es `mean squared error (mse)`
    - Como nuestro problema es una regresión la métrica de éxito es `mean absolut error (mae)`
    - Dado que la última capa del modelo NO es para clasificar, NO es necesario que tenga ninguna función de activación.
    - `input_dim` es una variante de `input_shape` la cual solo necesita especificar la canatidad de elementos en la entrada
    - de la red.

## Entrenamiento del modelo de regresión

3. **Cargando el dataset:**
    
    Este paso es básicamente el mismo que el de los ejemplos anteriores pero con el nuevo dataset de este ejemplo:
    ```python
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    print(train_data[0])
    print(train_targets[0])
    ```
    Respuesta esperada:
    ```commandline
    [  1.23247   0.        8.14      0.        0.538     6.142    91.7
       3.9769    4.      307.       21.      396.9      18.72   ]
    15.2
    ```

    Podemos conocer el valor de los 13 atributos de la casa y conocer que el precio de la misma es de 15.2 Mil dólares.

4. **Generando particiones de K-fold**

    El primer paso es definir el objeto `kf` de la clase `KFold`y pedirle que genere 5 `folds`

    ```python
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ```

    Adicionalmente, para que estas particiones sean replicables pondremos su `random_state` en un valor fijo.

    Ahora definimos una función auxiliar que nos permita dado unos vectores de valores de datos y de targets regrese las particiones de `x_train, x_test, y_train, y_test`:

    ```python
    def train_test_split_kf(xs: np.array, ys: np.array, train_size: np.array, test_size: np.array) -> np.array:
        x_train_ = xs[train_size]
        x_test_ = xs[test_size]
        y_train_ = ys[train_size]
        y_test_ = ys[test_size]
        return x_train_, x_test_, y_train_, y_test_
    ```

    Con la función definida podemos proceder a utilizar los rangos de ***train*** y ***test*** creados por el objeto `kf` y de ahí conseguir las particiones esperadas:
    
    ```python
    all_history = []  # Aquí guardaremos los resultados de cada fold
    
    for n_fold, (train, test) in enumerate(kf.split(train_data)):
        print(f"\t-I'm running fold {n_fold + 1}")
        x_train, x_test, y_train, y_test = train_test_split_kf(xs=train_data, ys=train_targets,
                                                               train_size=train, test_size=test)
    ```

***Nota***: La función kf.split(vector) toma un vector de entrada y regresa los indices de las particiones para ***train y test***


5. **Normalizando datos de entrenamiento**

Dado que estamos en un problema de regresión es MUY buena idea normalizar los datos, puesto que si tenemos una variable llamada: número de cuartos, quizá el rango de valores pueda ir de (1, 5) sin embargo, si tenemos otra que sea año de construcción cuyos valores esten entre (1930, 2022) podría aparentar que numéricamente hablando es más importante el año de construcción que el número de cuartos.
    
Para evitar estos inconvenientes lo que se hace es normalizar los datos, para ello a cada muestra del vector se le resta su promedio y se le divide entre su desviación standard. A este tipo de normalizado se le conoce como: `StandardScaler`

Primero: Creamos un objeto de la clase `StandarScaler`

```python
standard_scaler = StandardScaler()
```

A este objeto lo vamos a ***entrenar*** con la distribución de datos de nuestra partición de entrenamiento `x_train`:

```python
standard_scaler.fit(x_train)
```

Esto le permite al modelo aprender los valores de `mean` y `std` Ahora podemos proceder a normalizar los datos de entrenamiento y validación:

```python
x_train_s = standard_scaler.transform(x_train)
# Es MUY importante tener en cuenta que los datos de prueba NO LOS CONOZCO entonces NO tiene sentido obtener
# el promedio y desviación standard de una muestra que NO conozco por eso estoy normalizando con el promedio
# y std de la muestra a la que mi modelo sí tenía acceso mientras fue entrenada.
x_test_s = standard_scaler.transform(x_test)
```

6. **Entrenando el modelo**
    
Ahora podemos construir un modelo que sirva para las particiones de entrenamiento y validación de nuestros `fold` y ponerlo a entrenar por un número de épocas predefinido:

```python
model = build_model_regression(dim=13)
history = model.fit(x_train_s, y_train, epochs=n_epochs, batch_size=16,
                    validation_data=(x_test_s, y_test), verbose=0)
all_history.append(history.history)
```

## Análisis de resultados del modelo de regresión

7. Re estructurando datos de salida

Observemos los resultados de nuestros modelos que fueron guardados en `all_history`:

```python
out = all_history
print(out)
```
Respuesta esperada:
```commandline
[{'loss': [546.1715087890625, 396.9112548828125, 247.38885498046875, 133.14358520507812, 76.148681640625, ...], 
    'mae': [21.413293838500977, 17.873849868774414, 13.284331321716309, 8.88094425201416, 6.3826823234558105, ...], 
    'val_loss': [354.6539611816406, 238.64332580566406, 141.55055236816406, 78.4019546508789, 51.63487243652344, ...], 
    'val_mae': [17.48577117919922, 14.001570701599121, 10.338260650634766, 7.059571266174316, 5.430708885192871, ...]}, 
    
    {'loss': [508.8610534667969, 375.5047607421875, 231.6179656982422, 115.08528137207031, 63.184120178222656, ...], 
    'mae': [20.488609313964844, 16.794652938842773, 12.525876998901367, 8.449064254760742, 6.081153392791748, ...], 
    'val_loss': [424.28839111328125, 291.0716552734375, 168.71376037597656, 106.91384887695312, 76.11128234863281, ...], 
    'val_mae': [17.950767517089844, 14.14747142791748, 9.914441108703613, 7.694438934326172, 6.680331230163574]}, 
    
    ... 
    
    ] 
```
Esta estructura de diccionarios guardados en una lista NO es tan conveniente, lo que desearemos es un dataframe que contenga el promedio de los 5 `folds` sobre cada una de las n `epochs`eso lo podemos hacer de la siguiente manera:

```python
df = {}
for key in out[0].keys():
    row = []
    for fold in out:
        row.append(fold[key])
    row = np.array(row).mean(axis=0)
    df[key] = row
frame = pd.DataFrame(df)
frame = frame[offset:]
print(frame)
```

Respuesta :
```commandline
          loss       mae   val_loss   val_mae
5   39.485257  4.583744  36.213194  4.253143
6   30.369080  3.897518  30.883676  3.902345
7   25.733508  3.562452  26.227117  3.475329
8   22.751280  3.313163  25.029307  3.454233
9   20.649399  3.139169  23.158773  3.239379
10  18.837821  2.997525  21.098302  3.168812
```

8. Análisis de resultados final

Finalmente, podemos graficar los resultados de nuestro `frame` de resultados y observar los resultados de nuestro modelo con nuestra partición de ***test*** que hasta este momento NO habíamos utilizado.

```python
metric = "mae"
offset = 5
loss_values = frame['loss']
val_loss_values = frame['val_loss']
metric_values = frame[metric]
val_metric_values = frame[f"val_{metric}"]
epoch = range(1, len(loss_values) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
fig.suptitle("Neural Network's Result")
ax1.set_title("Loss function over epoch")
ax2.set_title(f"{metric} over epoch")
ax1.set(ylabel="loss", xlabel="epochs")
ax2.set(ylabel=metric, xlabel="epochs")
ax1.plot(epoch, loss_values, 'o-r', label='training')
ax1.plot(epoch, val_loss_values, '--', label='validation')
ax2.plot(epoch, metric_values, 'o-r', label='training')
ax2.plot(epoch, val_metric_values, '--', label='validation')
ax1.legend()
ax2.legend()
plt.savefig("imgs/results.png")
plt.close()
```

Respuesta:
![results.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fregresi%C3%B3n%2Fimgs%2Fresults.png)

Finalmente, evaluando la partición de pruebas:

```python
results = model.evaluate(test_data, test_targets)
```

Respuesta:
```commandline
4/4 [==============================] - 0s 1ms/step - loss: 960760.6875 - mae: 906.1437
```

ÉXITO TOTAL, ÉXITO ROTUNDO 

Hemos conseguido un error de menos de 1000 dólares en la predicción del precio de una nueva casa.

## Listado de referencias externas

* Training dataset (Wikipedia)  

[r000]: https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets#Training_data_set "referencia a training dataset en Wikipedia"

* Validation dataset (Wikipedia)  

[r001]: https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets#Validation_data_set "referencia a validation dataset en Wikipedia"

* Test dataset (Wikipedia)  

[r002]: https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets#Test_data_set "referencia a test dataset en Wikipedia"

* Normalización de datos (Microsoft learn)  

[r003]: https://learn.microsoft.com/es-es/azure/machine-learning/component-reference/normalize-data?view=azureml-api-2 "referencia a normalización de datos en Microsoft learn"

## Listado de imágenes

* Test dataset

[i000]: img/NNPKER032.gif "Test dataset"

* Tipos de variables

[i001]: https://i.imgur.com/I5mxbw3.png "Tipos de variables"

* Errores

[i002]: https://i.imgur.com/AqUQVz7.png "Errores"

* Errores

[i003]: https://i.imgur.com/AqUQVz7.png "Errores"


