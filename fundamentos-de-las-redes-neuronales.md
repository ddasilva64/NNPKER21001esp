# Fundamentos de la arquitectura de redes neuronales

<details>

<summary>Índice de contenidos</summary>

* [***La importancia de las redes neuronales en la actualidad***](#la-importancia-de-las-redes-neuronales-en-la-actualidad)
  * [***Concepto de red neuronal artificial***](#concepto-de-red-neuronal-artificial)  
  * [***Importancia de las redes neuronales artificiales***](#importancia-de-las-redes-neuronales-artificiales)

* [***Herramientas usadas en redes neuronales***](#herramientas-usadas-en-redes-neuronales)  
  * [***Keras***](#keras)  

* [***Deep Learning***](#deep-learning)  
  * [***Ciclo de ML vs DL***](#ciclo-de-ml-vs-dl)  
  * [***Problemas del DL***](#problemas-del-dl)
  
* [***Primera red neuronal con Keras***](#primera-red-neuronal-con-keras)  
  * [***Obtener datasets pre-cargados de keras***](#obtener-datasets-pre-cargados-de-keras) 
  * [***Shapes (formas) de los datos de entrenamiento y test del dataset de DL***](#shapes-formas-de-los-datos-de-entrenamiento-y-test-del-dataset-de-dl)
  * [***Modelo secuencial de 1 capa y multiples salidas***](#modelo-secuencial-de-1-capa-y-multiples-salidas)
  * [***Compilar el modelo con parámetros***](#compilar-el-modelo-con-parámetros)
  * [***Limpieza de datos***](#limpieza-de-datos)
  * [***Resumen de la configuración del modelo***](#resumen-de-la-configuración-del-modelo)  

* [***Entrenando el modelo de la primera red neuronal***](#entrenando-el-modelo-de-la-primera-red-neuronal)  
  * [***Entrenar la red neuronal***](#entrenar-la-red-neuronal)  
  * [***Evaluar el rendimiento de la red neuronal con datos de validación***](#evaluar-el-rendimiento-de-la-red-neuronal-con-datos-de-validación)
  * [***Guardar el modelo para usarlo después***](#guardar-el-modelo-para-usarlo-después)  

* [***La neurona artificial o perceptrón***](#la-neurona-artificial-o-perceptrón)  
  * [***Funcionamiento***](#funcionamiento)  
  * [***Componentes***](#componentes)
  * [***Acción del sesgo***](#acción-del-sesgo)
  * [***Neurona - función AND***](#neurona---función-and)  
  * [***Neurona - función XOR***](#neurona---función-xor)

* [***Arquitectura de una red neuronal***](#arquitectura-de-una-red-neuronal)  
  * [***Notas adicionales***](#notas-adicionales)  

* [***Funciones de activación***](#funciones-de-activación)  
  * [***Función Sigmoide***](#función-sigmoide)
  * [***Función Step (escalón de Heaviside)***](#función-step-escalón-de-heaviside)
  * [***Función ReLU (rectificadora)***](#función-relu-rectificadora)
  * [***Función Tanh (tangente hiperbólica)***](#función-tanh-tangente-hiperbólica)

* [***Funcion de pérdida (loss function)***](#funcion-de-pérdida-loss-function)  
  * [***Las 2 de las funciones de perdida más utilizadas***](#las-2-de-las-funciones-de-perdida-más-utilizadas)

* [***Descenso del gradiente***](#descenso-del-gradiente)  

* [***Minimización de la función de coste***](#minimización-de-la-función-de-coste)

* [***Backpropagation***](#backpropagation)  
  * [***Fases***](#fases) 

* [***Playground - Tensorflow***](#playground---tensorflow)  

</details>

## La importancia de las redes neuronales en la actualidad  

### Concepto de red neuronal artificial

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Red neuronal</span>][r000]***: Una red neuronal es un modelo de ***ML***, expresado mediante un programa, que se utiliza para la toma de decisiones de forma similar al cerebro humano, utilizando procesos que imitan la forma, en que las neuronas biológicas, trabajan juntas para identificar fenómenos, sopesar opciones y llegar a conclusiones.  

![Red neuronal][i000]  

***Las redes neuronales artificiales asisten a los humanos, en la toma de decisiones, con una asistencia humana limitada***. Aprenden y modelan las relaciones, no lineales y complejas, entre los datos de entrada y salida.

### Importancia de las redes neuronales artificiales

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Aplicaciones de las redes neuronales en la vida real</span>][r001]***:  

- Hacer generalizaciones y sacar conclusiones
- Comprender datos no estructurados y hacer observaciones generales sin un entrenamiento explícito. Por ejemplo, pueden reconocer que dos oraciones de entrada diferentes tienen un significado similar, por ejemplo:
  - ¿Puede explicarme cómo hacer el pago?
  - ¿Cómo puedo transferir el dinero?
- También, una red neuronal sería capaz de reconocer, en términos generales, que Baxter Road es un lugar, pero que Baxter Smith es nombre de persona.
- Identificación de tumores en imágenes médicas.
- etc.

![Aplicaciones de las redes neuronales en la vida real][i001]  

## Herramientas usadas en redes neuronales

### Keras

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Keras</span>][r002]***: Keras es una biblioteca de código abierto (con licencia MIT) escrita en Python, que se basa principalmente en el trabajo de François Chollet, un desarrollador de Google, en el marco del proyecto ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System). La primera versión de este software multiplataforma se lanzó el 27 de marzo de 2015. 

El objetivo de la biblioteca es acelerar la creación de redes neuronales. No funciona como un framework independiente, sino como una API que permite acceder a varios frameworks de aprendizaje automático y desarrollarlos. Entre los frameworks compatibles con Keras, se incluyen Theano, Microsoft Cognitive Toolkit (anteriormente CNTK) y TensorFlow.

![API Keras][i002]

En definitiva, Keras es una forma más amigable de acceder a frameworks independientes. A su vez, estos frameworks usan libererías de más bajo nivel para comunicarse directamente con el hardware del dispositivo para acceder y utilizar la GPU o a la CPU del ordenador. 

El código de Keras está alojado en [GitHub](https://es.wikipedia.org/wiki/GitHub) y existen foros y un canal de [Slack](https://es.wikipedia.org/wiki/Slack) de soporte.

## Deep Learning (DL)

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Deep Learning</span>][r003]***: Conjunto de algoritmos de ***ML*** que intenta modelar abstracciones de alto nivel, en datos usando arquitecturas computacionales, que admiten transformaciones no lineales múltiples e iterativas de datos expresados en forma matricial o tensorial.  

Es decir, ***DL*** es todo lo relacionado con las redes neuronales. Se llama aprendizaje profundo porque a mayor capas conectadas ente sí, se obtiene un aprendizaje más fino.

![Deep Learning][i003]

### Ciclo de ML vs DL

Los métodos tradicionales de ***ML*** tienen un parámetros de entrada que implican el uso de ***Feature Engineering***, la cual consiste en utilizar nuestro conocimiento sobre el negocio (business rules), para transformar los datos de entrada en formas más comprensibles para el modelo y poder trabajar con ellos.  

Los modelos basados en ***DL***, también necesitan una forma de convertir tipos de datos no estructurados en representaciones estructuradas. Sin embargo, en ***DL*** no tenemos que preocuparnos por limpiar y optimizar las variables de entrada, una vez definidas, puesto que serán las propias neuronas del modelo las que automáticamente permitirán decidir la importancia de cada una de nuestras variables de entrada (***Feature Importances***). Sin embargo, este tipo de estrategia conduce, también, a los **principales problemas del DL**.  

![Ciclo de ML vs DL][i004]
 
### Problemas del DL

* ***[<span style="font-family:Verdana; font-size:0.95em;color:red">Overfitting</span>][r005]***: Cuando el modelo “memoriza” los datos, en lugar de aprender, es decir, la red neuronal no sabe generalizar.

![Overfitting][i005]

* ***[<span style="font-family:Verdana; font-size:0.95em;color:red">Black box</span>][r006]***: Conocemos las entradas y las salidas de las redes neuronales, pero no conocemos lo que pasa en las capas intermedias de la red.

![Black box][i006]

## Primera red neuronal con Keras

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Dataset MNIST</span>][r007]***: El [dataset MNIST](http://yann.lecun.com/exdb/mnist/) es considerado el "Hello World" de la visión artificial. Contiene un conjunto de entrenamiento de 60.000 imágenes de dígitos manuscritos (de 0 a 9), y otro conjunto de pruebas con 10.000 muestras adicionales. Las imágenes originales fueron normalizadas de forma que cupiesen en un grid de 20x20 píxels manteniendo las proporciones de la imagen original, y el resultado se centró en un grid de 28x28 píxels. Es de estas modificaciones de donde proviene la "M" de "MNIST" (Modified National Institute of Standards and Technology).

![Dataset MNIST][i007]

Utilizaremos el dataset de MNIST .

```python
# Importación de librerías necesarias
from keras.datasets           import mnist
from keras                    import layers, models
from keras.utils              import to_categorical
import numpy                  as np
import matplotlib.pyplot      as plt
from google.colab             import drive                                      # mount drive    
```
```python
# Paths para guardar objetos
pathModel = '/content/drive/MyDrive/Colab Notebooks/NNPKER/ejemplos/Primera_red_neuronal_Keras/Modelo/'
PathImag  = '/content/drive/MyDrive/Colab Notebooks/NNPKER/ejemplos/Primera_red_neuronal_Keras/Imagenes/'
```

### Obtener datasets pre-cargados de Keras

```python
# Dataset
 (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
```

Keras cuenta con una sección de datasets para poder aprender ***DL***. Estos datasets ya cuentan con una función muy útil llamada ***load_data()*** la cuál permite cargar 4 particiones de los mismos:  

1. ***train_data***: Contiene datos (de entrada) para entrenar a el modelo.
2. ***train_labels***: Contiene la clasificación de cada uno de los datos de ***train_data***.
3. ***test_data***: Es una partición de datos con características similares a  ***train_data***, pero que el modelo de ***DL*** no ha conocido durante el proceso de entrenamiento (training).
4. ***test_labels***: Corresponde a la clasificación de los datos de ***test_data***.

### Shapes (formas) de los datos de entrenamiento y test del dataset de DL

En cualquier proyecto de ***ML*** (no solo de ***DL***)es indispensable conocer la distribución de los datos de entrenamiento (training) y prueba (testing), asi como conocer la forma (shape) de cada uno de los elementos que lo componen  

```python
# Shapes
print('Train data shape\t\t\t:',train_data.shape)
print('Train data example shape:', train_data[0].shape)
print('Train label shape:', train_labels.shape)
print('Train label example shape:', train_labels[0].shape)
print('Test data shape:', test_data.shape)
print('Test data example shape:', test_data[0].shape)
print('Test label shape:', test_labels.shape)
print('Test label example shape:', test_labels[0].shape)
```  

``` 
Train data shape            : (60000, 28, 28)
Train data instance shape   : (28, 28)
Train label shape           : (60000,)
Train label instance shape  : ()
Test data shape             : (10000, 28, 28)
Test data instance shape    : (28, 28)
Test label shape            : (10000,)
Test label instance shape   : ()
``` 

El dataset de entrenamiento está formado por 60.000 imágenes de 28x28 (***cada instancia es de 28x28***), mientras que el dataset de testing están formados por 10.000 imágenes (de las mismas dimensiones). Adicionalmente, vemos como las ***labels*** de ambos datasets ***son arreglos unidimencionales (arrays)***, ***solamente guardan el nombre de la clase***, en este caso un número, pero no la imagen (instancia).

```python
# Primera instancia del train_data
plt.imshow(train_data[0])
path = PathImag + 'numero_Primera_red_neuronal_con_Keras_train_data_0.png'
plt.savefig(path)
plt.imshow(train_data[0])
print(train_labels[0])
```
```
5
```
![Imagen de la primera instancia de train_data][i007a]  

```python
# Clase de la instancia train_data[0]
print('Class of train_dada instance[0]:',train_labels[0])
```
```
Class of train_dada instance[0]: 5
```

```python
# Primera instancia del test_data
plt.imshow(test_data[0])
path = PathImag + 'numero_Primera_red_neuronal_con_Keras_test_data_0.png'
plt.savefig(path)
plt.imshow(test_data[0])
print(test_labels[0])
```
```
7
```
![Imagen de la primera instancia de test_data][i007b]   

```python
# Clase de la instancia test_data[0]
print('Class of test_dada instance[0]:',test_labels[0])
```
```
Class of test_dada instance[0]: 7
```

```python
# Instancia 45 del train_data
path = PathImag + 'numero_Primera_red_neuronal_con_Keras_train_data_45.png'
plt.savefig(path)
plt.imshow(train_data[45])
print(train_labels[45])
```
```
9
```
![Imagen de la instancia 45 de train_data][i007c]  

```python
# Clase de la instancia train_data[45]
print('Class of train_dada instance[45]:',train_labels[45])
```
```
Class of train_dada instance[45]: 9
```
```python
# Instancia 45 del test_data
path = PathImag + 'numero_Primera_red_neuronal_con_Keras_test_data_45.png'
plt.savefig(path)
plt.imshow(test_data[45])
print(test_labels[45])
```
```
5
```
![Imagen de la instancia 45 de test_data][i007d]   

```python
# Clase de la instancia test_data[45]
print('Class of test_dada instance[45]:',test_labels[45])
```
```
Class of test_dada instance[45]: 5
```

### Modelo secuencial de 1 capa y multiples salidas

```python
# Arquitectura del modelo de DL para la clasificación de los números decimales
#   Funciones de activación ReLu y SotfMax
model = models.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=(28*28, )))
model.add(layers.Dense(10, activation="softmax"))
```

* ***input_shape=(28\*28, )***: Indica que la entrada de la red neuronal tendré 784 (28x28) neuronas, una para cada pixel de la imagen.
* ***layerse.Dense(10, )***: El 10 indica que queremos que la última capa, la de clasificación, clasifique entre 10 posibles clases (números del 0 al 9).

### Compilar el modelo con parámetros

A pesar de que ya se ha creado la arquitectura del modelo, aún es necesario indicar ciertos parámetros que modificarán la forma en que la red neuronal es entrenada (***optimizer***, ***loss***, ***metrics***). 

- ***optimizer***: Algoritmo matemático que será utilizado para cambiar la distribución de pesos y bias (sesgos) en las neuronas. La red va aprendiendo tras cada iteración.
- ***loss***: Forma de definir lo cerca o no, que estamos del objetivo a optimizar.
- ***metrics***: Evalua el rendimiento de la red, tanto en el training set como en el testing set.

```python
# Compilación del modelo
#   optimizer = distribución de pesos y bias (sesgos) en las neuronas
#   loss      = lo cerca o no, que estamos del objetivo a optimizar
#   metrics   = rendimiento de la red (train & test)
model.compile(optimizer ='rmsprop',
              loss      ='categorical_crossentropy',
              metrics   =['accuracy'])
```

### Limpieza de datos

En el momento de crear la arquitectura de la red, la entrada de información tiene que ser un vector unidimensional (vector de imágenes), sin embargo, las imágenes tienen 2 dimensiones y por eso debemos transformar la forma del ***train_data*** a una forma que tenga 60.000 imágenes, cada una de las cuales, ***debe ser un vector de 1 dimensión con 784 valores (28x28)***.

Las imágenes, del dataset que usamos, están codificadas en 8 bits (escala de grises), eso significa que la cantidad de niveles de gris posibles son $2^8 = 256$. Van de 0 (negro) a 255 (blanco). 

Si las imágenes fuesen de 16 bits, en escala de grises (por ejemplo, las imágenes médicas de mamografías), se tendría $2^{16} = 65536$ niveles de gris, que irían de 0 (negro) a 65535 (blanco).

Como las imágenes de nuestro dataset tienen 8 bits, podemos normalizar los datos de entrada dividiendo entre el número más grande de esta forma y pasamos de una escala de [0, 255] a una de [0, 1], debido a que las redes neuronales trabajan más cómodamente con números decimales que con enteros. 

```python
# Limpieza de datos
#   Un píxel tiene de 0 a 255 en escala de grises
#   si dividimos por 255 cada pixel será un decimal, entre 0 y 1
x_train = train_data.reshape((60000,28*28))
x_train = x_train.astype('float32')/255

x_test = test_data.reshape((10000,28*28))
x_test = x_test.astype('float32')/255

print(x_train[0].shape)
```
```
(784,)
```

```python
# Transformación de las clases en categorías
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
```
```python
# Clase de la primera instancia
train_labels[0]
```
```python
# Primera instancia (de 0 a 1)
y_train[0]
```
```
array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
```

### Resumen de la configuración del modelo

Algo muy útil en ***DL*** es poder ver un resumen de la arquitectura de la red neuronal.

```python
# Resumen de la configuración del modelo
print(model.summary())
```
```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 512)                 │         401,920 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           5,130 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 407,050 (1.55 MB)
 Trainable params: 407,050 (1.55 MB)
 Non-trainable params: 0 (0.00 B)
None
```

## Entrenando el modelo de la primera red neuronal

### Entrenar la red neuronal

Ahora que los datos de entrenamiento y testing tienen un mejor formato, podemos entrenar a nuestra red neuronal. 

* ***epochs***: Nº de iteraciones que queremos que el modelo realice para ser optimizando.  
* ***batch_size***: Parámetro que indica cuando el dataset es muy grande (60.000). Es mejor ir entrenando el modelo de forma paralela, con conjuntos más pequeños (mejor 128 instancias que 60.000, al mismo tiempo).

```python
# Entrenar la red neuronal
model.fit(x_train, y_train, epochs=5, batch_size=128)
```
```
Epoch 1/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - accuracy: 0.8776 - loss: 0.4335
Epoch 2/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9662 - loss: 0.1158
Epoch 3/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9794 - loss: 0.0702
Epoch 4/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9841 - loss: 0.0508
Epoch 5/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9890 - loss: 0.0392
<keras.src.callbacks.history.History at 0x7e033663bdc0>
```

***Nota***: En la última iteración el accuracy (exactitud) es del 98,90% (muy bueno).

### Evaluar el rendimiento de la red neuronal con datos de test

Ahora que hemos visto que el accuracy (exactitud) del modelo, sobre los datos de entrenamiento, es de 98,90%, vamos a ponerlo a prueba con los datos de testing, que el modelo no ha visto mientras era entrenado.

```python
# Evaluar el rendimiento de la red neuronal con datos de test
model.evaluate(x_test, y_test)
```
```
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.9742 - loss: 0.0802
[0.06863386929035187, 0.9782999753952026]
```

El modelo ha obtenido ***0,9783 (97,83%)*** de accuracy sobre un dataset desconocido, lo cual está muy bien.

### Guardar el modelo para usarlo después

Finalmente, como paso extra, podemos exportar el modelo con todos los pesos, arquitectura y su compilación para cargarlo después y ponerlo a clasificar lo que nosotros queramos.

```python
# Guardar el modelo para usarlo después
path = pathModel + 'ModeloMNIST.h5'
model.save(path)
```

Ahora en un archivo diferente podemos cargar el modelo y ponerlo a clasificar:

```python
# ibrerías necesarias
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models             import load_model
from keras.datasets           import mnist
from keras.utils              import to_categorical
from google.colab             import drive   
```

```python
# Path en que hemos guardado el modelo
drive.mount('/content/drive')
pathModel = '/content/drive/MyDrive/Colab Notebooks/NNPKER/ejemplos/Primera_red_neuronal_Keras/Modelo/'
```

```python
path = pathModel + 'ModeloMNIST.h5'
print(path)
model = load_model(path)
(_, _), (test_data, test_labels) = mnist.load_data()
x_test = test_data.reshape((10000, 28*28))
x_test = x_test.astype("float32")/255
y_test = to_categorical(test_labels)
model.evaluate(x_test, y_test)
```
```
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
/content/drive/MyDrive/Colab Notebooks/NNPKER/ejemplos/Primera_red_neuronal_Keras/Modelo/ModeloMNIST.h5
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9742 - loss: 0.0802
[0.06863386929035187, 0.9782999753952026]
```

## La neurona artificial o perceptrón

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Perceptrón</span>][r008]***: *Es el modelo matemático más simple de una neurona, tanto biológica como artificial*. Una neurona sola y aislada carece de razón de ser. La neurona artificial (también llamada ***[<span style="font-family:Verdana; font-size:0.95em;color:red">perceptrón</span>][r008]***), fue concebida por ***[<span style="font-family:Verdana; font-size:0.95em;color:red">Frank Rosenblatt</span>][r008f]***, la década de los 50’s del siglo pasado.

![Frank Rosenblatt][i008f]  

![Neurona vs perceptrón][i008h] 

### Funcionamiento

![Perceptrón][i008]  

1. ***Suma ponderada de las entradas, con sus pesos*** (weights w) = salida lineal
2. ***Función de activación*** = no linealidades
3. ¿El modelo no satisface de forma adecuada el problema?
  a. NO => Se itera, actualizando los pesos, hasta resolver el problema
  b. SI => Fin

### Componentes

- ***Variables de entrada***: $x_{1}...x_{n}$
- ***Pesos asignados a las variables de entrada***: $Wx_{1}...Wx_{n}$
- ***Suma ponderada de la multiplicación, de las entradas y los pesos***: $x_{1}·Wx_{1}...x_{n}·Wx_{n}$ = SP
- ***Suma del sesgo (bias)***: SP + Wb (puede o no tener peso asignado)

### Acción del sesgo

![Neurona - bias][i008a]  

El ***bias*** de la neurona le ***permite*** tener más ***elasticidad***. Por ejemplo, la función $2x^2$ cuando X=0 entonces $f(x)=0$. Si le añadimos un ***bias*** b, entonces ***la función se desplaza b unidades***.

### Neurona - función AND

Las neuronas nos permiten solucionar problemas, por ejemplo, con una correcta distribución de pesos, bias y de la elecciçón de la función de activación adecuada, podemos crear una puerta lógica AND. 

***Tabla de verdad de la puerta lógica AND***  

|A|B|A AND B|
|-|-|:-:|
|1|0|0|
|0|1|0|
|0|0|0|
|1|1|1|

Podemos asignar, por ejemplo, los valores

$Wx_{1}=2$
$Wx_{2}=1$
$Bx=-3$

Ahora, escogeremos una ***función de activación***, que permita ***deformar*** la salida lineal (tenemos que eliminar linealidades).  

En nuestro ejemplo escogemos una ***función*** de tipo ***escalón***, para la ***función de activación***.

Concretamente, la función que escogemos es 

$$
\begin{align}
0 &=> w·x-u\le 0\\
1\ & \text{en otro caso}
\end{align}
$$

![Neurona - escalón][i008c] 

de modo que si $f(x)$ es negativo o 0, entonces la salida será 0. De otro modo será 1.  

Tabla de verdad de la función AND

|A|B|$\Sigma$|Función de activación|Salida|
|-|-|:-:|:-:|:-:|
|1|0|1·2+0·1-3|f(-1)=0|0|
|0|1|0·2+1·1-3|f(-2)=0|0|
|0|0|0·2+0·1-3|f(-3)=0|0|
|1|1|1·2+1·1-3|f(0)=1|1|

***Con lo cual queda demostrado (ccqd), que el comportamiento de la neurona es igual a una puerta lógica AND***.

![Neurona - AND][i008b]  

![Neurona - función de activación AND][i008g]  

### Neurona - función XOR

***Tabla de verdad de la puerta lógica XOR***  

|A|B|A XOR B|
|-|-|:-:|
|1|0|1|
|0|1|1|
|0|0|0|
|1|1|0|

La puerta ***XOR (u OR exclusiva) representa la función de la desigualdad, es decir, la salida es verdadera, si y solo si las entradas no son iguales***.

Existen muchos problemas que no se pueden solucionar con una sola neurona, tal es el caso de la función XOR:  

![Neurona - XOR][i008d]  

Observemos que ***la región de soluciones ciertas (1), se separa de las demás soluciones mediante dos líneas***, mientras que en la puerta AND, se separaba por una sola línea.  

Un ***problema linealmente separable*** es aquél que ***puede dividirse en dos áreas claramente diferenciadas mediante una línea***. Vale cualquier línea para separarlos, tan compleja como se desee.

Esta operación tan sencilla marca ya una barrera en las redes neuronales, que no se romperá hasta que en 1986 Rumelhart y McClelland diseñaran el ***perceptrón multicapa*** (o ***MLP***, multilayer perceptron), revolucionando todo lo conocido hasta entonces en redes neuronales.

![Neurona - 2XOR][i008e]

[***Ver desarrollo RNA de la puerta lógica XOR***](slides%20y%20notas\rna_xor.md)

***Corolario***

***A mayor cantidad de neuronas, mayor posibilidad de solucionar problemas más complejos***. Esto no es 100% cierto, para el 100% de los casos.  

Existen problemas que, con una menor cantidad de neuronas, se pueden resolver de forma más eficiente, pero, ***en general***, tener mayor cantidad de neuronas, facilita la resolución de problemas. Este es el caso de la puerta lógica XOR. 

## Arquitectura de una red neuronal

La arquitectura de la red puede ser dividida en tres partes:

- ***Capa de entrada***: Lugar en donde los datos son introducidos.
- ***Capas ocultas***: Se encuentran entre la capa de entrada y la de salida y son las que hacen las operaciones matemáticas. Pueden haber 1 o más capas ocultas.
- ***Capa de salida***: La que hace la predicción.

Dentro de la arquitectura de la red neuronal se realizan muchos ***productos punto***, entre las entradas de cada perceptrón y sus respectivos pesos. ***Estas operaciones son lineales***.

Las ***funciones de activación*** son la solución, al colapso de las linealidades en las capas de la red neuronal. Es decir, ***introducen no linealidades***.

![Neurona - 2XOR][i000]

> ### Notas adicionales

- Las capas más cercanas a la de entrada, obtienen las características más generales del problema, luego ***cuanto más profunda sea la capa, más específica es la característica aprendida***.
- ***Las últimas capas tienden más al overfitting***.

Las redes neuronales tienen m variables de entrada y cada capa tendrá n neuronas. A cada variable de entrada le corresponderán m pesos.

Imaginemos que tenemos m=4, n=3

$
\begin{equation}
  \begin{pmatrix}
  x_{1} \\
  x_{2} \\
  x_{3} \\
  x_{4} \\
  \end{pmatrix} ·
  \begin{pmatrix}
  w_{1} & w_{2} & w_{3} & w_{4}  \\
  w_{1} & w_{2} & w_{3} & w_{4} \\
  w_{1} & w_{2} & w_{3} & w_{4}  \\
  \end{pmatrix} = 
  \begin{pmatrix}
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4}  \\
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4}  \\
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4}  \\
  \end{pmatrix}
\end{equation}
$

![m variables de entrada][i009]

![multiplicaciones m x n][i010]

Para obtener los valores en cada neurona de la segunda capa, también, se realizará el producto del vector de m variables, por la matriz de n neuronas (***producto punto***).

Al final (de todas las capas), se le sumará el sesgo.

$
\begin{equation}
  \begin{pmatrix}
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4}  \\
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4}  \\
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4}  \\
  \end{pmatrix} +  b =
  \begin{pmatrix}
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4} + b \\
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4} + b  \\
  w_{1}·x_{1} + w_{2}·x_{2} + w_{3}·x_{3} + w_{4}·x_{4} + b  \\
  \end{pmatrix}
\end{equation}
$

![Es necesario añadir el bias][i011]

Generalizando, todas las neuronas se multiplican por todas las variables de la capa anterior, para obtener nuevos valores. Es decir, se hará un producto matricial (m x n)

$
\begin{equation}
  \begin{pmatrix}
  x_{1} \\
  \vdots \\
  x_{m} \\
  \end{pmatrix} ·
  \begin{pmatrix}
  w_{1} & \cdots & w_{n}  \\
  \vdots  & \ddots  & \vdots \\
  w_{1} & \cdots & w_{n}  \\
  \end{pmatrix} =
  \begin{pmatrix}
  w_{1}·x_{1} + & \cdots & + w_{n}·x_{m} + b \\
  \vdots  & \ddots  & \vdots \\
  w_{1}·x_{1} +  & \cdots & + w_{n}·x_{m} + b  \\
  \end{pmatrix}
\end{equation}
$

Sin embargo, hasta este preciso momento aún hay un área de oportunidad enorme:

![Área de oportunidad][i012]

Si apilamos varías capas de redes neuronales, todas y cada una de ellas, a su salida, tienen una función lineal. ***La suma de varias funciones lineales es otra función lineal***, lo cal no tiene mucho sentido porque entonces toda la información (de en medio), pierde funcionalidad. La solución es que ***las salidas de estas capas intermedias no sean lineales***. Para ello usamos las ***funciones de activación***.

## Funciones de activación

Las ***funciones de activación*** permiten ***eliminar la linealidad de las salidas de las neuronas***. 

Estas funciones pueden ser ***discretas*** (tener un conjunto finito de valores decimales) o ***continuas*** (estar dentro de un intervalo de valores decimales).

![Algunas funciones de activación][i013]

***Saber qué función de activación utilizar en cada momento***

![Qué función de activación utilizar en cada momento][i014]

La función activación a utilizar en cada momento va a depender del tipo de problema y siempre se puede jugar con las activaciones de cada capa. Sin embargo, es normalmente aceptado que las ***capas ocultas*** utilicen [***ReLU***](#función-relu-rectificadora) y ***la última dependa de si el problema es de clasificación binaria o multiple***.  

Por lo general, si el problema tiene ***más de dos clases*** se utilizará ***softmax*** o [***sigmoide***](#función-sigmoide), por otro lado, si es ***binaria*** puede ser [***sigmoide***](#función-sigmoide) o ***step***, finalmente, si el problema es una ***regresión*** entonces basta con usar una ***linear***.

### Función Sigmoide

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

La ***función de activación sigmoide***, a menudo representada como $\sigma(x)$, es una función infinitamente diferenciable e históricamente importante en el desarrollo de las redes neuronales. La función de activación sigmoide se expresa como:

$f(x) = \frac{1}{(1 + e^{-x})}$

Esta función toma una entrada real y la reduce a un valor en el intervalo entre 0 y 1. Tiene una curva en forma de "S", que tiende a 0 para los números negativos grandes y a 1 para los números positivos grandes.  

***Los resultados pueden interpretarse como probabilidades, lo que la hace natural para los problemas de clasificación binaria***.

***Nota***: Los problemas, en los que la ***columna objetivo es categórica***, se denominan problemas de ***clasificación***. Los ***problemas de clasificación binaria tienen dos categorías posibles(sí o no)***, mientras que los problemas de ***clasificación multiclase*** tienen ***más de dos categorías posibles***.

![Sigmoide][i015]

Las sigmoides fueron populares en las primeras RNA (Redes Neuronales Artificiales), ya que ***el gradiente es más fuerte, cuando la salida es cercana a 0,5***, lo que permite un entrenamiento eficaz por ***retropropagación***. Sin embargo, sufren el problema de "***desvanecimiento de gradiente***", que ***dificulta el aprendizaje en las redes neuronales profundas***.

***Nota***: ***A medida que los valores de entrada se vuelven significativamente positivos o negativos, la función se satura en 0 o 1***, con una pendiente extremadamente plana. En estas regiones, el gradiente es muy próximo a cero. Esto da lugar a cambios muy pequeños en los pesos durante la [***retropropagación***](#backpropagation), sobre todo para las neuronas de las primeras capas de las redes profundas, lo que hace que el ***aprendizaje sea penosamente lento o incluso lo detenga***. Esto es el ***problema de desvanecimiento de gradiente*** en las redes neuronales.

El ***principal caso de uso*** de la ***función sigmoide*** es como ***activación de la capa de salida de los modelos de clasificación binaria***.  

### Función Step (escalón de Heaviside)

```python
def step(x):
    return np.piecewise(x, [x < 0.0, x > 0.0], [0, 1])
```

![Step][i016]

### Función ReLU (rectificadora)

```python
def relu(x):
    return np.maximum(0, x)
```

La ***función de activación de unidad lineal rectificada (ReLU)*** tiene la forma:

$f(x) = max(0, x)$

Umbraliza la entrada en cero, ***devuelve 0 para valores negativos y la propia entrada para valores positivos***.

***Para entradas mayores que 0, ReLU actúa como una función lineal con un gradiente de 1***. Esto significa que no modifica la escala de las entradas positivas y permite que el gradiente pase sin cambios durante la ***retropropagación***. Esta propiedad es fundamental para ***mitigar el problema de desvanecimiento de gradiente***.

Aunque ***ReLU*** es lineal para la mitad de su espacio de entrada, ***técnicamente es una función no lineal*** porque tiene un punto no diferenciable en x=0, donde cambia bruscamente con respecto a x. ***Esta no linealidad permite a las redes neuronales aprender patrones complejos***.

Como ***ReLU*** produce cero para todas las entradas negativas, conduce naturalmente a activaciones dispersas; ***en un momento dado, solo se activa un subconjunto de neuronas, lo que conduce a una computación más eficiente***.

La función ReLU ***es computacionalmente poco costosa porque implica un umbral simple en cero***. Esto ***permite a las redes escalar a muchas capas sin un aumento significativo de la carga computacional***, en comparación con funciones más complejas como la función ***tanh (tangente hiperbólica)*** o la ***sigmoide***.

![ReLu][i017]

### Función Tanh (tangente hiperbólica)

```python
def tanh(x):
    return np.tanh(x)
```

La ***función de activación de tangente hiperbólica (tanh)***, se define como:

$f(x) = \frac{(e^{x} - e^{-x})}{(e^{x} + e^{-x})}$

Esta función produce valores en el intervalo de -1 a +1. Eso significa que puede tratar valores negativos con más eficacia que la función ***sigmoide***, que tiene un intervalo de 0 a 1.

A diferencia de la función ***sigmoide***, ***tanh*** está centrada en cero, lo que significa que su ***resultado es simétrico alrededor del origen del sistema de coordenadas***. Esto suele considerarse una ***ventaja porque puede ayudar a que el algoritmo de aprendizaje converja más rápidamente***.

Como el resultado de ***tanh*** oscila entre -1 y +1, tiene gradientes más fuertes que la función ***sigmoide***. ***Los gradientes más fuertes suelen dar lugar a un aprendizaje y una convergencia más rápidos durante el entrenamiento***, porque tienden a ser ***más resistentes frente al problema de desvanecimiento de gradiente***, que los gradientes de la función ***sigmoide***.

A pesar de estas ventajas, ***la función tanh sigue sufriendo el problema de desvanecimiento de gradiente***. Durante la ***retropropagación***, los gradientes de la función ***tanh*** pueden llegar a ser muy pequeños (próximos a cero). Esta cuestión es especialmente problemática para las redes profundas con muchas capas; los gradientes de la ***función de pérdida*** pueden llegar a ser demasiado pequeños para realizar cambios significativos en los pesos durante el entrenamiento, ya que se retropropagan a las capas iniciales. Esto puede ***ralentizar drásticamente el proceso de entrenamiento y provocar una mala convergencia***.

La función ***tanh*** se utiliza con frecuencia en las capas ocultas de una red neuronal. Debido a su naturaleza centrada en cero, ***cuando los datos también se normalizan para que tengan media cero, puede resultar un entrenamiento más eficiente***.

***Si hay que elegir entre la función sigmoide y la función tanh*** y no se tiene ninguna razón específica para preferir una a la otra, ***tanh suele ser la mejor opción*** por las razones antes mencionadas. Sin embargo, la decisión también puede verse influida por el caso de uso concreto y el comportamiento de la red durante los experimentos de entrenamiento iniciales.

![Tangente hiperbólica (Tanh)][i018]

## Funcion de pérdida (loss function)

Una ***RNA*** tiene como finalidad generar una ***predicción***, ya sea de una ***regresion*** (un valor continuo) o una ***clasificación*** (clases pre-definidas). Entonces, ¿cómo sabemos si la predicción ha sido buena o mala, o cómo de lejos está del valor real?. Para responder a esta pregunta, las redes neuronales necesitan tener valores conocidos, para tener un marco de referencia.

En el ejemplo pasado veíamos, que el dataset de MNIST tiene dos grandes grupos: 

- **Training**
  - data training
  - label training  

- **Testing**
  - data testing
  - label testing

Los ***datos son los valores de entrada*** y las ***labels son los*** valores de salida o ***valores a predecir***. Es en este conjunto de valores (dataset de salida), encontramos ejemplos de las respuestas correctas que, dado los valores de entrada, deben llegar a la salida esperada (label). 

El ***objetivo de la RNA*** es ***utilizar los pesos y los bias*** (de las capas del modelo), ***para generar una salida***, la cual sea ***lo más similar posible a las labels esperadas*** y no solo a una, sino ***a todas las de los datasets de training y testing***.

Es aquí cuando entra en juego el concepto de ***loss function*** (funcion de pérdida), la cual permite ***comparar lo lejos está la RNA de los valores reales***.  

De acuerdo al tipo de problema existen diferentes ***loss functions***, pero todas ***tienen como objetivo*** permitir observar ***lo buena que es la RNA, para predecir la variable de salida***. Así, ***para mejorarla, podemos ir actualizando los pesos de las neuronas para que vayan reduciendo el coste***. 

### Las 2 de las funciones de perdida más utilizadas

**MSE (Mean Squared Error)**

Esta función de perdida está diseñada ***para problemas de regresión***, dónde queremos obtener un ***valor continuo***, como ***por ejemplo, el valor de una casa***.   

![MSE][i019]

La función ***MSE*** toma el ***cuadrado de la distancia entre el valor real y la predicción***, para ***castigar con más fuerza a los valores más alejados de la predicción***.  

Implementación de la función de ***MSE*** en python:

```python
import numpy as np

def mse(y: np.array, y_hat: np.array, derivative: bool = False):
    if derivative:
        return y_hat - y
    else:
        return np.mean((y_hat - y)**2)

if __name__ == '__main__':
    real = np.array([0, 0, 1, 1])
    prediction = np.array([0.9, 0.5, 0.2, 0])
    print(mse(real, prediction))

```
```
0.675
```

**Cross Entropy (entropía cruzada)**

Esta ***función de perdida*** está diseñada ***para problemas de clasificación***. Mide la ***distancia entre la predicción del algoritmo y el valor real, para cada una de las clases del problema***.

![Cross entropy][i020]

En este ejemplo, el valor real a predecir era el círculo marcado como su representación [***one hot encoding***](https://interactivechaos.com/es/manual/tutorial-de-machine-learning/one-hot-encoding) como *(1, 0, 0)* y la predicción del algoritmo fue *(0.5, 0.3, 0.2)*, entonces la fórmula de ***Cross Entropy*** toma en cuenta el valor real $p(x)$ y el logaritmo de la predicción $log(q(x))$. Entonces

Designaremos a la predicción como ***$\hat{y}$*** y al valor real como ***$y$***.  

Si $p$ y $q$ son variables discretas:

$$\mathrm {H} (p,q)=-\sum _{x}p(x)\,\log q(x)\!$$

```Python
# Implementación de la pérdida de entropía cruzada entre los valores predichos y los verdaderos 
#   de las etiquetas de clase
#     Entradas: Valores predichos, valores verdaderos
#     Salida: La pérdida de entropía cruzada entre ellos


# Importar biblioteca requerida
import torch.nn as nn
import torch

# Función Cross Entropy
def cross_entropy(y_pred, y_true):

	# Cálculo de valores softmax para valores predichos
	y_pred = softmax(y_pred)
	loss = 0
	
	# Pérdida de entropía cruzada
	for i in range(len(y_pred)):

    # La pérdida se calcula con la fórmula matemática anterior
		loss = loss + (-1 * y_true[i]*np.log(y_pred[i]))

	return loss

# y_true: Distribución de probabilidad verdadera
y_true = [1, 0, 0, 0, 0]

# y_pred: Valores predichos para cada clase
y_pred = [10, 5, 3, 1, 4]

# Llama a la función cross_entropy
cross_entropy_loss = cross_entropy(y_pred, y_true)

print("Pérdida de entropía cruzada: ", cross_entropy_loss)

```

## Descenso del gradiente

Matemáticamente hablando, ***las funciones continuas, también son derivables***. Intuitivamente, se puede decir que, una función es continua cuando en su gráfica no aparecen saltos o cuando el trazo de la gráfica no tiene "huecos".

![Descenso del gradiente][i021a]

Podemos decir que (a) y (b) no son continuas, pero (c) sí lo es. Entonces, ***que una función sea continua permite que sea diferenciable***, y a su vez, esto quiere decir que, ***dado un punto que pertenezca a la función f(x) se puede encontrar una recta tangente a dicho punto*** (una recta que únicamente va a tocar en un solo punto a dicha función).

![Descenso del gradiente][i021b]

**Recta tangente a la función en un punto** 

Imaginemos que tenemos la siguiente función:

![Descenso del gradiente][i021c]

Interesa encontrar en qué puntos, la función tiene un ***valor máximo y un valor mínimo***, en otras palabras optimizarla. Podemos usar la derivada de la función para ir obteniendo la recta pendiente en todos y cada uno de los puntos de la función en un rango establecido. La ***pendiente*** a su vez tendrá un cierto grado de inclinación, este puede ser una ***inclinación positiva o negativa***, de acuerdo a si el punto de la función se encuentra en la parte ***creciente o decreciente*** de la misma. 

Sin embargo, únicamente ***en los puntos máximos y mínimos la pendiente de dicho punto tendrá una inclinación de 0***.

![Descenso del gradiente][i021d]

En otras palabras, ***la derivada permite encontrar los valores máximos y mínimos locales de una función***. Esto pasará ***cuando la derivada se haga 0***.

Esto es muy ***útil*** porque lo que buscamos, con las ***funciones de perdida***, es ***optimizarlas acercándolas lo más posible a 0***. Dicho de otra forma, ***lo que buscamos con las funciones de perdida es que tengan la menor perdida posible y es justo por eso que utilizamos las derivadas de las funciones, para irnos acercando poco a poco al objetivo*** de optimización. 

***El algoritmo del gradiente descendiente (gradient descend) tiene como objetivo, dada una función, encontrar donde se encuentra su mínimo local***.

![Descenso del gradiente y mínimo local][i021e]

En este escenario hay un término muy importante conocido como ***learning rate*** (velocidad de aprendizaje), es decir, el ***tamaño del salto*** que el algoritmo de ***gradient descend*** tiene permiso dar, a cada iteración. 

![Descenso del gradiente y learning rate][i021f]

Si él ***learning rate*** es ***muy pequeño***, puede ***tardar muchos saltos en llegar al punto de optimización*** deseado. Pero si él ***learning rate*** es ***demasiado grande*** entonces este tamaño de salto ***no va a permitir converger en ningún punto***. 

Idealmente, debemos tener un ***learning rate lo suficientemente pequeño para que asegure que podemos converger, pero lo suficientemente grande para tomar la menor cantidad de iteraciones***. En la actualidad existen métodos de optimización que no necesitan especificar este parámetro por defecto. Un ***learning rate*** utilizado con bastante frecuencia es: *0,001*  

En la actualidad hay una gran variedad de optimizadores y los más nuevos tienden a ser más eficientes y veloces para encontrar el mínimo de una función.

![Descenso del gradiente][i021g]

Finalmente, para encontrar la dirección de la derivada es necesario entender el concepto de derivadas parciales:

![Descenso del gradiente][i021h]

***La derivada parcial de una función de varias variables, es la derivada con respecto a cada una de esas variables manteniendo las otras como constantes***. En definitiva, ***en una red neuronal tenemos funciones de muchísimas variables, y cada una de ellas expresa una dimensión***. Entonces, ***para encontrar el descenso del gradiente, debemos obtener la derivada para cada una de las dimensiones del problema***. 

En realidad, ***la derivada va a darnos la dirección en la que crece la función***, pero si queremos obtener la dirección en la que decrece, basta con que la multipliquemos por -1. 

## Minimización de la función de coste

***Todo algoritmo de ML se configura durante su entrenamiento***.  

Una regresión lineal de la forma

$$ax + b = 0$$  

se va a configurar determinando los valores de $a$ y de $b$ (parámetros que la definen). 

***La función de coste nos medirá la bondad del modelo final***.  

Consideremos, por ejemplo, el error cuadrático medio:

$$MSE=\frac{1}{n}·\Sigma (y_{n}-\hat {y})^{2}$$  

el cual nos dice que el error global que va a cometer el modelo vendrá dado por el valor medio de los cuadrados de las diferencias entre las predicciones y los valores reales. Siguiendo con el ejemplo de regresión lineal, para cada par de valores de $a$ y $b$ que puedan determinarse durante el entrenamiento tendremos un error cuadrático medio diferente.  

***El objetivo del entrenamiento es encontrar los valores de $a$ y $b$ que minimicen la función de coste***. Pero, ¿cómo encontramos los valores de los parámetros ($a$ y $b$) que minimizan esa función de coste?. ***Si la función de coste es sencilla y el conjunto de datos a entrenar no es excesivamente grande, podemos recurrir al cálculo infinitesimal para el cálculo del mínimo de la función***. De hecho, en el caso de una función lineal (como la comentada), hay una fórmula (la llamada "***ecuación normal***") que nos ***devuelve los valores de los parámetros que minimizan la función***.

El problema es que, ***normalmente***, ***el cálculo de los parámetros que minimizan*** una cierta ***función de coste aplicando cálculo infinitesimal puede resultar computacionalmente inabordable***. En una red neuronal, por ejemplo, podemos tener cientos de miles de pesos y de sesgos, elementos cuyos valores hay que determinar durante el entrenamiento de la red neuronal para que ésta pueda realizar predicciones precisas. ***Es por ello que se suele recurrir a algoritmos como el descenso de gradiente para el cálculo de estos mínimos***. Pero, por supuesto, aplicar el algoritmo de ***descenso de gradiente*** implica el cálculo del gradiente de la ***función de coste*** en un punto concreto de su dominio (en un punto determinado por unos parámetros concretos), lo que a su vez implica el ***cálculo de la derivada parcial de la función de coste con respecto a todos y cada uno de los parámetros que van a determinar el funcionamiento de la red***.

Imaginemos una red neuronal de decenas de capas ocultas, con miles o decenas de miles de neuronas artificiales en cada una de ellas, escojamos una neurona cualquiera de una capa cualquiera, y uno de los enlaces que le llegan desde una neurona de la capa anterior. Ese enlace viene caracterizado por un peso que, en mayor o menor grado, va a determinar el comportamiento de la red. Y minimizar el error de la red (minimizar la función de coste) implica determinar el valor correcto de ese peso (y del resto de miles de pesos y bias comentados). ¿Cuál es la derivada parcial de la función de coste con respecto a dicho peso? Ese peso va a influir en el resultado devuelto por la neurona que habíamos escogido, lo que influirá a su vez en los resultados que devolverán las neuronas de la siguiente capa... El cálculo de la derivada parcial de la función de coste con respecto a dicho peso no parece una tarea fácil.

***En el pasado se utilizaban métodos que, basados en la fuerza bruta, estimaban la derivada parcial en cuestión, modificando el valor del peso y viendo el efecto que tenía en la salida de la red, pero esto, además de ser poco fiable, era computacionalmente muy caro***.

## Backpropagation

***Backpropagation*** es un proceso, que es especialmente importante, ya que, ***a medida que se entrena una red neuronal, los nodos de las capas intermedias son capaces de organizarse por sí mismos***. De esta manera, ***cada uno de estos nodos son capaces de aprender a reconocer distintas características de los datos de entrada***.

Gracias al método de ***backpropagation*** las redes neuronales son capaces de ***identificar patrones de datos incompletos o arbitrarios*** y encontrar la solución más adecuada para el problema que se les haya planteado, ya que serán capaces de hallar un patrón similar a las características que hayan aprendido a reconocer durante su entrenamiento. Es decir, ***este algoritmo sirve para detectar errores en procesos que implican el uso de redes neuronales***.

### Fases

El entrenamiento de las redes neuronales es un proceso complejo que implica distintas etapas. El método de backpropagation es la cuarta etapa del proceso y, al mismo tiempo se compone de distintas fases:

- ***Elección de la entrada y de la salida***: Este es el primer paso en el funcionamiento del algoritmo y es el momento en el que se determina una entrada para todo el proceso de retropropagación, desde el punto de entrada hasta la salida deseada.

- ***Configuración***: Una vez configurados los valores de entrada y de salida, el algoritmo procede a asignar una serie de valores secundarios que le permiten modificar parámetros dentro de cada capa y nodo que conforman la red neuronal.

- ***Cálculo de error***: En este paso se determina el error total, a partir del análisis de los nodos y capas de red neuronal.  

- ***Minimización de errores***: Una vez detectados los errores, el algoritmo procede a minimizar su efecto en el conjunto de la red neuronal.

- ***Actualización de parámetros***: Si la tasa de error es muy alta, el método de bakcpropagation, ajusta y actualiza los parámetros para reducirla lo máximo posible.

- ***Modelado para la predicción***: Tras la optimización de los errores, el método de cálculo de backpropagation, evalúa las entradas de prueba adecuadas para garantizar que se obtienen el resultado deseado.

![Backpropagation][i022a]

Hasta este momento entendemos que para llegar a obtener la predicción, las salidas de las capas anteriores, en una red, funcionan como las entradas de la siguiente capa y así hasta llegar a la última capa, que será la encargada de dar la predicción final.

[Animación backpropagation con MNIST][i022b]

Sin embargo, el error calculado por la ***cost function*** (función de coste), viene dado únicamente por la respuesta de la última capa y una vez que se tiene este error, la última capa puede argumentar que ella no tuvo la culpa, sino que el error viene de la capa anterior y así sucesivamente. 

Entonces, haciendo uso de derivadas parciales, podemos ir hacia atrás, capa a capa, distribuyendo los cambios necesarios para ir disminuyendo el error, esto es en sí la forma más fácil de entender la ***backpropagation***.

## Playground - Tensorflow

***Tensorflow*** nos ofrece una ***herramienta sumamente útil*** que nos provee de un entorno gráfico para probar conceptos de redes neuronales en problemas de ***clasificación y regresión***: [Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.87931&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


![Backpropagation][i023a]

**Playground - Tensorflow**

Nos permite tener acceso de forma simple e inmediata a los siguientes parámetros configurables:

- Learning rate
- Activation function
- Regularization
- Regularization rate
- Problem type

En la izquierda de la pantalla podemos seleccionar la distribución de datos, así como la cantidad de datos utilizada para entrenar y evaluar al modelo.

En el centro de la pantalla obtenemos la arquitectura de la red, podemos configurar las entradas de datos que deseamos, la cantidad de capas ocultas y la cantidad de neuronas en cada capa.

En la derecha de la pantalla obtenemos el test loss y un gráfico que representa la clasificación de la red.

***Problema del desvanecimiento del gradiente (vanishing gradient problem)***  

En ***ML***, el problema de desvanecimiento de gradiente es una dificultad encontrada al entrenar redes neuronales artificiales mediante métodos de aprendizaje basados en descenso estocástico de gradientes y de ***retropropagación***. En tales métodos, cada uno de los pesos de la red neuronal recibe una actualización proporcional a la derivada parcial de la función de error con respecto al peso actual en cada iteración de entrenamiento.

El problema es que, ***en algunos casos, el gradiente se irá desvaneciendose a valores muy pequeños, impidiendo cambiar su valor eficazmente, al peso***. ***En el peor de los casos, esto puede impedir que la red neuronal continúe su entrenamiento***. Como ejemplo de la causa del problema, funciones de activación tradicionales como la función de la tangente hiperbólica tienen gradientes en la gama (-1, 1), y la retropropagación computa gradientes por la regla de la cadena. Esto tiene el efecto de multiplicar n de estos números pequeños para computar gradientes de las "capas" de frente en una red de n capas, significando que el gradiente (señal de error) disminuye exponencialmente con n mientras las capas de frente se entrenan muy despacio.

La retropropagación permitió, a los investigadores entrenar redes neuronales supervisadas profundas, desde un inicio con muy poco éxito. La tesis de diploma de 1991 de Hochreiter identificó formalmente la razón de este fracaso en el "problema de desvanecimiento de gradiente", lo cual no sólo afecta a las redes prealimentadas de muchas capas, sino también a las redes recurrentes. Estas últimas se entrenan por desdoblamiento en redes neuronales prealimentadas muy profundas, donde se crea una capa nueva cada vez que se da un paso en la secuencia de entrada por la red.

Cuando se usan funciones de activación cuyas derivadas pueden tomar valores más grandes, uno de los riesgos es encontrar el denominado ***problema de gradiente explosivo***.

## Listado de referencias externas

* Red neuronal (Wikipedia)  

[r000]: https://es.wikipedia.org/wiki/Red_neuronal_artificial "referencia a red neuronal en Wikipedia"

* Aplicaciones de las redes neuronales en la vida real (Wikipedia)  

[r001]: https://es.wikipedia.org/wiki/Red_neuronal_artificial#Aplicaciones_de_la_vida_real "referencia a aplicaciones de las redes neuronales en la vida real en Wikipedia"

* Keras (Wikipedia)  

[r002]: https://es.wikipedia.org/wiki/Keras "referencia a Keras en Wikipedia"

* Deep learning (Wikipedia)  

[r003]: https://es.wikipedia.org/wiki/Aprendizaje_profundo "referencia a deep learning en Wikipedia"

* Overfitting (Wikipedia)  

[r005]: https://es.wikipedia.org/wiki/Sobreajuste "referencia a overfitting en Wikipedia"

* Black box (Wikipedia)  

[r006]: https://es.wikipedia.org/wiki/Caja_negra_(sistemas) "referencia a black box en Wikipedia"

* Dataset MNIST (InteractiveChaos)

[r007]: https://interactivechaos.com/es/manual/tutorial-de-deep-learning/el-dataset-mnist "referencia a dataset MNIST en InteractiveChaos"

* Perceptrón (Wikipedia)

[r008]: https://es.wikipedia.org/wiki/Perceptr%C3%B3n "referencia a perceptrón en Wikipedia"

* Frank Rosenblatt (Wikipedia)

[r008f]: https://es.wikipedia.org/wiki/Frank_Rosenblatt "referencia a Frank Rosenblatt en Wikipedia"

## Listado de imágenes

* Red neuronal  

[i000]: https://i.imgur.com/BG1freF.png "Red neuronal"

* Aplicaciones de las redes neuronales en la vida real  

[i001]: https://i.imgur.com/oZPwqPH.png "Aplicaciones de las redes neuronales en la vida real"

* API Keras  

[i002]: https://i.imgur.com/o2L4zb8.png "API Keras"

* Deep learning  

[i003]: https://i.imgur.com/zC0JPvm.png "Deep learning"

* Ciclo de ML vs DL

[i004]: https://i.imgur.com/q4ojoTz.png "Ciclo de ML vs DL"

* Overfitting

[i005]: https://i.imgur.com/16ePu0k.png "Overfitting"

* Black box

[i006]: https://i.imgur.com/hZ3WkdY.png "Black box"

* Dataset MNIST

[i007]: https://i.imgur.com/HJNRkYx.png "Dataset MNIST"

* Imagen de la primera instancia de train_data

[i007a]: https://i.imgur.com/jLMc73b.png "Imagen de la primera instancia de train_data"

* Imagen de la primera instancia de test_data

[i007b]: https://i.imgur.com/BVGvp4f.png "Imagen de la primera instancia de test_data"

* Imagen de la instancia 45 de train_data

[i007c]: https://i.imgur.com/vXvgDTW.png "Imagen de la instancia 45 de train_data"

* Imagen de la instancia 45 de test_data

[i007d]: https://i.imgur.com/0iw645T.png "Imagen de la instancia 45 de test_data"

* Perceptrón

[i008]: https://i.imgur.com/VPcdqkS.png "Perceptrón"

* Neurona - bias

[i008a]: https://i.imgur.com/Xp5i8TH.png "Neurona - bias"

* Neurona - AND

[i008b]: https://i.imgur.com/HAkboR6.png "Neurona - AND"

* Neurona - escalón

[i008c]: https://i.imgur.com/IVEdqKq.png "Neurona - escalón"

* Neurona - XOR

[i008d]: https://i.imgur.com/vZdtEnp.png "Neurona - XOR"

* Neurona - 2XOR

[i008e]: https://i.imgur.com/E4ReQc3.png "Neurona - 2XOR"

* Frank Rosenblatt

[i008f]: https://i.imgur.com/TCTGgZJ.png "Frank Rosenblatt"

* Neurona - función de activación AND 

[i008g]: https://i.imgur.com/W0wWvb5.png "Neurona - función de activación AND"

* Neurona vs perceptrón  

[i008h]: https://i.imgur.com/zX1sXz9.png "Neurona vs perceptrón"

* Salida neurona - XOR  

[i008i]: https://i.imgur.com/FFC6bb2.png "Salida neurona - XOR"

* m variables de entrada 

[i009]: https://i.imgur.com/DDdRIFj.png "m variables de entrada"

* Multiplicaciones es m x na 

[i010]: https://i.imgur.com/RYPDEgi.png "Multiplicaciones es m x n"

* Es necesario añadir el bias 

[i011]: https://i.imgur.com/DR1nfTf.png "Es necesario añadir el bias"

* Área de oportunidad 

[i012]: https://i.imgur.com/GCtyKmq.png "Área de oportunidad"

* Algunas funciones de activación 

[i013]: https://i.imgur.com/kOg3GxA.png "Algunas funciones de activación"

* Qué función de activación utilizar en cada momento 

[i014]: https://i.imgur.com/XUuUX0F.png "Qué función de activación utilizar en cada momento"

* Sigmoide 

[i015]: https://i.imgur.com/3QTNpHN.png "Sigmoide"

* Step 

[i016]: https://i.imgur.com/WdJ1Daq.png "Step"

* ReLu 

[i017]: https://i.imgur.com/2ZTyyuS.png "ReLu"

* Tangente hiperbólica (Tanh) 

[i018]: https://i.imgur.com/mYMnGig.png "Tangente hiperbólica (Tanh)"

* MSE 

[i019]: https://i.imgur.com/mh02I4m.png "MSE"

* Cross entropy 

[i020]: https://i.imgur.com/039aZGA.png "Cross entropy"

* Descenso del gradiente

[i021a]: https://i.imgur.com/QZWuGDw.png "Descenso del gradiente"

* Descenso del gradiente

[i021b]: https://i.imgur.com/lCQmX2o.png "Descenso del gradiente"

* Descenso del gradiente

[i021c]: https://i.imgur.com/trLwDa9.png "Descenso del gradiente"

* Descenso del gradiente

[i021d]: https://i.imgur.com/AGZ22Tj.png "Descenso del gradiente"

* Descenso del gradiente y mínimo local

[i021e]: img/NNPKER022e.gif "Descenso del gradiente y mínimo local"

* Descenso del gradiente y learning rate

[i021f]: img/NNPKER022f.gif "Descenso del gradiente y learning rate"

* Descenso del gradiente

[i021g]: img/NNPKER022g.gif "Descenso del gradiente"

* Descenso del gradiente

[i021h]: https://i.imgur.com/1wSM1CQ.png "Descenso del gradiente"

* Backpropagation

[i022a]: https://i.imgur.com/vU2fdG3.png "Backpropagation"

* Backpropagation

[i022b]: img/NNPKER023b.gif "Backpropagation"

* Playground - Tensorflow

[i023a]: https://i.imgur.com/ABlmCCT.png "Playground - Tensorflow"

