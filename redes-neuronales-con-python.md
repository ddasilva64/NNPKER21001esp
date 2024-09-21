# Redes neuronales con Python

<details>

<summary>Índice de contenidos</summary>

* [***Dimensiones y tensores***](#dimensiones-y-tensores)   

* [***Crear una RNA usando NumPy***](#crear-una-rna-usando-numpy)
  * [***Creación de un dataset***](#creación-de-un-dataset)
  * [***Definición de las funciones de activación***](#definición-de-las-funciones-de-activación)
  * [***Función de pérdida***](#función-de-pérdida)
  * [***Función de inicialización de pesos***](#función-de-inicialización-de-pesos)

* [***Entrenamiento forward***](#entrenamiento-forward)
  * [***Forward propagation***](#forward-propagation)

* [***Aplicando backpropagation y descenso del gradiente***](#aplicando-backpropagation-y-descenso-del-gradiente)
  * [***Backpropagation***](#backpropagation)
  * [***Gradient descent***](#gradient-descent)

* [***Entrenamiento y análisis de resultados de la RNA***](#entrenamiento-y-análisis-de-resultados-de-la-rna)
  * [***Train model function***](#train-model-function)
  * [***Definir arquitectura de la RNA***](#definir-arquitectura-de-la-rna)
  * [***Entrenamiento del modelo***](#entrenamiento-del-modelo)
  * [***Probando el modelo con datos nuevos***](#probando-el-modelo-con-datos-nuevos)

</details>

## Dimensiones y tensores

***[<span style="font-family:Verdana; font-size:0.95em;color:red">NumPy</span>][r000]***: ***NumPy*** es una biblioteca para el lenguaje de programación ***Python*** que da soporte para crear ***vectores y matrices grandes multidimensionales***, junto con una gran colección de funciones matemáticas de alto nivel para operar con ellas. El precursor de ***NumPy***, Numeric, fue creado originalmente por Jim Hugunin con contribuciones de varios otros desarrolladores. En 2005, Travis Oliphant creó ***NumPy*** incorporando características de la competencia Numarray en Numeric, con amplias modificaciones. ***NumPy*** es un software de código abierto y cuenta con muchos colaboradores.

![Travis Oliphant][i000]  

![NumPy][i001]  

***Python*** nos ofrece una amplia gama de tipos de datos y dimensiones, con ayuda de ***NumPy***, como:

- ***Escalares***: Números. Dimensión = 0
- ***Vectores***: Listas de números. Dimensión = 1.
- ***Matrices***: Matrices (de 2 dimensiones) de números (dataframes de Pandas, por ejemplo). Dimensión = 2.
- ***Tensores***: Conjuntos de números de más de 2 dimensiones. Puede ser una o más matrices de 3 dimensiones.

![Dimensiones en Python ][i002]  

Python:
```python
# Importamos NumPy
import numpy as np

# Función para obtener las características de un objeto
def funcFeatures(obj, name):

  dimension = obj.ndim
  tipo      = ''

  if (dimension == 0):
    tipo = 'escalar'
  elif (dimension == 1):
    tipo = 'vector'
  elif (dimension == 2):
    tipo = 'matriz'
  elif (dimension >= 3):
    tipo = 'tensor'

  print(' -------------------------------\n','Caracteríscas del objeto: ' + name, '\n\tTipo de objeto:\t', tipo, "\n\tValor/es:\t", obj, "\n\tShape:\t\t", obj.shape, "\n\tDimensión/ones:\t", obj.ndim)

# Definición de los objetos
obj01 = np.array(37)
obj02 = np.array([2, 3, 5, 7, 11])
obj03 = np.array([[2 , 3 , 5 , 7 , 11],
                  [13, 17, 19, 23, 29]])
obj04 = np.array([[[2 , 3 , 5 , 7 , 11],
                   [13, 17, 19, 23, 29]],
                  [[2 , 4 , 6 , 8 , 10],
                   [12, 14, 16, 18, 20]],
                  [[1 , 3 , 5 , 7 , 9 ],
                   [11, 13, 15, 17, 19]]])
obj05 = x = np.array([[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7]])

# Obtención de las características de los objetos
funcFeatures(obj01, 'obj01')
print()
funcFeatures(obj02, 'obj02')
print()
funcFeatures(obj03, 'obj03')
print()
funcFeatures(obj04, 'obj04')
print()
funcFeatures(obj05, 'obj05 - original -')
print()
funcFeatures(obj05.reshape(8, 1), 'obj05 - reshape(8, 1) -')
print()
funcFeatures(obj05.reshape(2, 4), 'obj05 - reshape(2, 4) -')
print()
funcFeatures(obj05.reshape(2, 4), 'obj05 - reshape(2, 4) -')
print()
funcFeatures(obj05, 'obj05 - original -')
print()
funcFeatures(obj05.T, 'obj05 - traspuesta -')
```

Resultados:
```commandline
 -------------------------------
 Caracteríscas del objeto: obj01 
	Tipo de objeto:	 escalar 
	Valor/es:	 37 
	Shape:		 () 
	Dimensión/ones:	 0

 -------------------------------
 Caracteríscas del objeto: obj02 
	Tipo de objeto:	 vector 
	Valor/es:	 [ 2  3  5  7 11] 
	Shape:		 (5,) 
	Dimensión/ones:	 1

 -------------------------------
 Caracteríscas del objeto: obj03 
	Tipo de objeto:	 matriz 
	Valor/es:	 [[ 2  3  5  7 11]
 [13 17 19 23 29]] 
	Shape:		 (2, 5) 
	Dimensión/ones:	 2

 -------------------------------
 Caracteríscas del objeto: obj04 
	Tipo de objeto:	 tensor 
	Valor/es:	 [[[ 2  3  5  7 11]
  [13 17 19 23 29]]

 [[ 2  4  6  8 10]
  [12 14 16 18 20]]

 [[ 1  3  5  7  9]
  [11 13 15 17 19]]] 
	Shape:		 (3, 2, 5) 
	Dimensión/ones:	 3

 -------------------------------
 Caracteríscas del objeto: obj05 - original - 
	Tipo de objeto:	 matriz 
	Valor/es:	 [[0 1]
 [2 3]
 [4 5]
 [6 7]] 
	Shape:		 (4, 2) 
	Dimensión/ones:	 2

 -------------------------------
 Caracteríscas del objeto: obj05 - reshape(8, 1) - 
	Tipo de objeto:	 matriz 
	Valor/es:	 [[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]] 
	Shape:		 (8, 1) 
	Dimensión/ones:	 2

 -------------------------------
 Caracteríscas del objeto: obj05 - reshape(2, 4) - 
	Tipo de objeto:	 matriz 
	Valor/es:	 [[0 1 2 3]
 [4 5 6 7]] 
	Shape:		 (2, 4) 
	Dimensión/ones:	 2

 -------------------------------
 Caracteríscas del objeto: obj05 - reshape(2, 4) - 
	Tipo de objeto:	 matriz 
	Valor/es:	 [[0 1 2 3]
 [4 5 6 7]] 
	Shape:		 (2, 4) 
	Dimensión/ones:	 2

 -------------------------------
 Caracteríscas del objeto: obj05 - original - 
	Tipo de objeto:	 matriz 
	Valor/es:	 [[0 1]
 [2 3]
 [4 5]
 [6 7]] 
	Shape:		 (4, 2) 
	Dimensión/ones:	 2

 -------------------------------
 Caracteríscas del objeto: obj05 - traspuesta - 
	Tipo de objeto:	 matriz 
	Valor/es:	 [[0 2 4 6]
 [1 3 5 7]] 
	Shape:		 (2, 4) 
	Dimensión/ones:	 2
```

[Ver código anterior aquí](https://github.com/ddasilva64/NNPKER21001esp/tree/master/ejemplos/Dimensiones_y_tensores.ipynb)

## Crear una RNA usando NumPy

Vamos a implementar una ***RNA*** desde cero, sin utilizar ningún framework, con ***NumPy***.  

### Creación de un dataset

Para este primer paso vamos a crear un dataset de 2 dimensiones, lo cual nos permitirá hacer la gráfica y ver fácilmente la distribución de clases en el mismo.

El dataset lo crearemos utilizando la función `make_gaussian_quantiles` de [sklear.datasets](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html)

Características del dataset:

- ***Cantidad de muestras***: n_samples = 1000
- ***Cantidad de características de entrada***: n_features = 2
- ***Cantidad de clases a predecir***: n_classes = 2

```python
# Importación de las librerías que necesitamos
from   sklearn.datasets   import make_gaussian_quantiles
import matplotlib.pyplot  as plt
import numpy              as np

# Creación del dataset
gaussian_quantiles = make_gaussian_quantiles(mean=None,
                                             cov=0.1,
                                             n_samples=1000,
                                             n_features=2,
                                             n_classes=2,
                                             shuffle=True,
                                             random_state=None)

X, Y = gaussian_quantiles

# Necesario para hacer el plot más cómodo
Y = Y[:,np.newaxis]

# Shapes de X e Y de 2 clases
print('Shape de las entradas     de la red:', X.shape)
print('Shape de las predicciones de la red:', Y.shape)

# Gráfico de dispersión (scatter plot), de la distribución de los datos
plt.title("Problema de clasificación")
plt.scatter(X[:,0], X[:,1], c=Y[:,0], s=40, cmap=plt.cm.Spectral)
```

Resultados:
```commandline
Shape de las entradas     de la red: (1000, 2)
Shape de las predicciones de la red: (1000, 1)
```

![Problema de clasificación][i003]  

Tenemos 1.000 muestras de coordenadas (x,y), por eso las variables de entrada tienen el Shape de (1000, 2), mientras que tenemos un vector de labels Y que solo tienen dos clases (0, 1). Esta es la razón por la que la variable de salida tiene un Shape de (1000, 1).

### Definición de las funciones de activación

Definimos las funciones Sigmoide y ReLU.

```python
def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
    else:
        return 1 / (1 + np.exp(-x))


def relu(x, derivate=False):
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(0, x)
```

Utilizaremos la función ***ReLU***, como función de activación para las capas ocultas y la función ***Sigmoide***, como la última función de activación para la capa de clasificación. Lo cual, es una configuración muy habitual.

### Función de pérdida

Usaremos ***MSE*** como función de perdida para la red neuronal:

```python
# Función de pérdida (MSE)
def mse(y, y_hat, derivate=False):
    if derivate:
        return 2*(y_hat - y)
    else:
        return np.mean((y_hat - y) ** 2)
```

### Función de inicialización de pesos

Cada capa de una ***RNA*** está definida por una serie de pesos (W) y de sesgos (b). Cuando creamos una ***RNA*** debemos empezar definiendo estos valores con un valor por defecto aleatorio. Posteriormente, los procesos de optimización iran mejorando estos pesos y sesgos aleatorios.

***Nota***: Todo el código, de aquí en adelante, está optimizado para trabajar con n layers. De forma automática, el ***forward*** y el ***backward propagation*** se adaptan a la cantidad de layers.

```python
def initialize_parameters_deep(layer_dims: list) -> dict:
    """
    Genera un diccionario de pesos y sesgos para una RNA, de acuerdo a su arquitectura de capas
      :param layer_dims: lista que representa la cantidad de neuronas presente en cada capa de la red
      :return: dict: parameters. 
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(0, L - 1):
        """
        Multiplicar por 2 y restar 1 es una forma de normalizar los datos para que vayan 
        de -1 a 1, de esta forma encajan mejor con la distribución de datos de entrada 
        de nuestro problema, pero tampoco es indispensable
        """
        parameters[f'W{l + 1}'] = (np.random.rand(layer_dims[l], layer_dims[l + 1]) * 2) - 1  

        parameters[f'b{l + 1}'] = (np.random.rand(1, layer_dims[l + 1]) * 2) - 1
        
        print(f"Inicializando PESO W{l + 1} con dimensiones:", parameters[f'W{l + 1}'].sahpe)
        print(f"Inicializando BIAS b{l + 1} con dimensiones:", parameters[f'b{l + 1}'].sahpe)

    return parameters
```

## Entrenamiento forward

### Forward propagation

Definición de variables:

- $X$: Variables de entrada del modelo
- $Y$: Etiquetas reales de las clases a predecir
- $W_{i}$: Pesos de la i-ésima capa.
- $b_{i}$: Sesgos de la i-ésima capa.
- $Z_{i}$: Parte lineal del proceso de la ***RNA*** np.dot($X$, $W$) + $b$ de la i-ésima capa.
- $A_{i}$: Función de activación aplicada a $Z$ de la i-ésima capa.
- $y\_hat$: Predicción final de la ***RNA***, correspondiente a $A$ de la última capa.
- $d(variable)_{i}$: Representa la derivada de cierta variable. Por ejemplo, $dW_{3}$ corresponde a la derivada de los pesos de la capa 3.

Programamos un paso de nuestra función de forward propagation:

```python
def linear_forward(A, W, b):
    Z = np.dot(A, W) + b
    return Z

def linear_activation_forward(A_prev, W, b, activation_function):
    Z = linear_forward(A_prev, W, b)
    A = activation_function(Z)
    return A

def forward_step(A0, params, activations_functions, n_layers):
    L = n_layers
    params["A0"] = A0
    for i in range(1, L + 1):
        params[f"A{i}"] = linear_activation_forward(params[f"A{i - 1}"], params[f"W{i}"], params[f"b{i}"],
                                                    activations_functions[i])
    y_hat = params[f"A{L}"]
    return y_hat
```

***Notas***:
- La primera capa de la red corresponde a la entrada de las varibles a clasificar, a su vez, podemos llamar a estos datos como:
  - La función `linear_forward` necesita los ***pesos*** y ***bias*** (sesgos) del ***layer*** (capa) actual y la respuesta de la ***función de activación*** del ***layer*** (capa) anterior.
  - En la ***capa oculta*** 1, la respuesta A anterior corresponde con la entrada de datos. Solamente es en este caso.

## Aplicando backpropagation y descenso del gradiente

### Backpropagation

En el proceso de ***backpropagation*** el primer paso es obtener el error entre el valor real Y y el valor predicho por la red y_hat. Una vez que se calculan las derivadas de Z y W de la última capa, entonces podemos ir para atrás calculando las otras dZ y dW para las capas anteriores.

```python
def backpropagation(Y, y_hat, params, activations_functions, error_function, n_layers):
    L = n_layers
    params[f'dZ{L}'] = error_function(Y, y_hat, True) * activations_functions[L](params[f'A{L}'], True)
    params[f'dW{L}'] = np.dot(params[f'A{L - 1}'].T, params[f'dZ{L}'])

    for l in reversed(range(2, L + 1)):
        params[f'dZ{l - 1}'] = np.matmul(params[f'dZ{l}'], params[f'W{l}'].T) * activations_functions[l - 1](
            params[f'A{l - 1}'], True)

    for l in reversed(range(1, L)):
        params[f'dW{l}'] = np.matmul(params[f'A{l - 1}'].T, params[f'dZ{l}'])

    return params
```

### Gradient descent

El último paso es,con las derivadas ya calculadas, actualizar los pesos $W_{i}$ y los bias $b_{i}$ de cada capa, utilizando las derivadas calculadas en el punto anterior y un ***learning rate*** (lr).

```python
def gradient_descent(params, lr, n_layers):
    L = n_layers

    for l in reversed(range(1, L + 1)):
        params[f'W{l}'] = params[f'W{l}'] - params[f'dW{l}'] * lr
        params[f'b{l}'] = params[f'b{l}'] - (np.mean(params[f'dZ{l}'], axis=0, keepdims=True)) * lr

    return params
```

## Entrenamiento y análisis de resultados de la RNA

### Train model function

Ahora, con todas las funciones que hemos definido anteriormente, podemos crear una nueva función que sirva como gestora de funciones y que permita poner, de forma secuencial, todos los pasos del entrenamiento de la red, para una cantidad de iteraciones definidas (`epochs`):

```python
def train_model(X, Y, layer_dims, params, activations_functions, error_function, lr, epochs):
    errors = []
    n_layers = len(layer_dims) - 1
    j = 1
    for _ in range(epochs):
        y_hat = forward_step(X, params, activations_functions, n_layers)
        params = backpropagation(Y, y_hat, params, activations_functions, error_function, n_layers)
        params = gradient_descent(params, lr, n_layers)

        if _ % 100 == 0:
            e = error_function(Y, y_hat)
            if _ % 1000 == 0:
                print(j, "error:", e)
                j += 1
            errors.append(e)
            
    return errors, params
```

### Definir arquitectura de la RNA

En este punto, definimos todas las variables que definen la arquitectura de la red como: 
- El número de capas  
- El número de neuronas por capa  
- El learning rate  
- Las funciones de activaciones en cada capa oculta  
- La cantidad de epochs  


```python
layer_dims = [2, 4, 8, 1]
lr = 0.002
activations_functions = [0, relu, relu, sigmoid]
params = initialize_parameters_deep(layer_dims)
epochs = 10000
```

Retorno:
```commandline
Inicializando PESO W1 con dimensiones: (2, 4)
Inicializando BIAS b1 con dimensiones: (1, 4)
Inicializando PESO W2 con dimensiones: (4, 8)
Inicializando BIAS b2 con dimensiones: (1, 8)
Inicializando PESO W3 con dimensiones: (8, 1)
Inicializando BIAS b3 con dimensiones: (1, 1)
```

### Entrenamiento del modelo

Ya tenemos todo lista para ejecutarse la función `train_model`, durante n `epochs`, utilizando `mse` como nuestra ***función de pérdida***.

```python
errors, params = train_model(X, Y, layer_dims, params, activations_functions, mse, lr, epochs)
plt.plot(errors)
plt.title("MSE over epochs")
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.savefig("imgs/model.png")
plt.close()
```

Retorno:
```commandline
1 error: 0.24757298982437223
2 error: 0.05393229428625825
3 error: 0.059482476810252996
4 error: 0.05376995780762481
5 error: 0.045345930433615116
6 error: 0.03746680657241106
7 error: 0.03602928839608933
8 error: 0.03626453751967989
9 error: 0.031568468850135964
10 error: 0.022516339724916422
```

![Modelo][i004]  

### Probando el modelo con datos nuevos

Finalmente, podemos crear un nuevo dataset aleatorio de prueba, que tenga una distribución similar al dataset de entrenamiento y observar como reliza la clasificación.

```python
data_test = (np.random.rand(1000, 2) * 2) - 1
prediction = forward_step(data_test, params, activations_functions, 3)
y = np.where(prediction >= 0.5, 1, 0)
plt.scatter(data_test[:, 0], data_test[:, 1], c=y[:, 0], s=40)
plt.title("NN prediction")
plt.savefig("imgs/prediction.png")
plt.close()
```

Retorno:  
![Predicción][i005]  


## Listado de referencias externas

* NumPy (Wikipedia)  

[r000]: https://es.wikipedia.org/wiki/NumPy "NumPy en Wikipedia"

## Listado de imágenes

* Travis Oliphant  

[i000]: https://i.imgur.com/FIKN9NU.png "Travis Oliphant"

* Logo NumPy  

[i001]: https://i.imgur.com/os3e46i.png "Logo NumPy"

* Dimensiones en Python  

[i002]: https://i.imgur.com/yRMPuhx.png "Dimensiones en Python"

* Problema de clasificación  

[i003]: https://i.imgur.com/3XpqFfX.png "Problema de clasificación"

* Modelo  

[i004]: https://i.imgur.com/9nDZObG.png "Modelo"

* Predicción  

[i005]: https://i.imgur.com/OBksWuo.png "Predicción"