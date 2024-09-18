# Construcción de una Red Neuronal Artificial (RNA) para resolver el problema XOR

## Definición del problema XOR

El problema que se propone es construir una RNA que genere la tabla de verdad de de la puerta lógica ***XOR (o exclusivo)***, la cual es una función lógica que devuelve verdadero, si y solo si las entradas son diferentes. Las entradas y salidas son:

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


Utilizaremos una RNA, con dos neuronas en la capa oculta, utilizando la ***función de activación ReLU (Rectified Linear Unit)*** y la ***función sigmoide*** en la capa de salida.

## Estructura de la red neuronal

- **Capa de entrada**: 2 neuronas (una para cada entrada).
- **Capa oculta**: 2 neuronas con ***función de activación ReLU***.
- **Capa de salida**: 1 neurona con ***función de activación sigmoide***.

## Pesos y sesgos

Denotamos los pesos y sesgos de la siguiente manera:

- Pesos de la capa oculta: 

$$ W_{h} = 
      \begin{pmatrix} 
      wh_{11} & wh_{12} \\
      wh_{21} & wh_{22} 
   \end{pmatrix}
$$

- Sesgos de la capa oculta: 

$$b_{h} = 
   \begin{pmatrix} 
      bh_{11} \\ 
      bh_{21} 
   \end{pmatrix}
$$

- Pesos de la capa de salida: 

$$W_{e} = 
   \begin{pmatrix} 
      we_{11} & we_{12} 
   \end{pmatrix}
$$

- Sesgo de la capa de salida: 

$$b_{e}$$

Posteriormente, asignaremos valores (configuración).

## Propagación hacia adelante

### Capa oculta

Usamos dos neuronas en la capa oculta con la ***función de activación ReLU***.  

Para una entrada  

$$x = 
   \begin{pmatrix} 
      x_{11} \\ 
      x_{21} 
   \end{pmatrix}
$$

, la salida de la capa oculta se calcula como:

$$ h = 
   \text{ReLU}(W_{h}x+b_{h}) 
$$

Donde la función ***ReLU*** se define como:  

$$ \text{ReLU}(z) = 
   \max(0, z) 
$$

Donde 

- $z_{1}=wh_{11} \cdot x_{11} + wh_{12} \cdot x_{21} + bh_{11}$
- $z_{2}=wh_{21} \cdot x_{11} + wh_{22} \cdot x_{21} + bh_{21}$

Es decir,

- ***Neurona 1***: 

$$h_{1} =
   ReLU(z_{1}) = \max(0, z_{1})
$$

- ***Neurona 2***: 

$$h_{2} = 
   ReLU(z_{2}) = \max(0, z_{2})
$$

### Capa de salida

La salida de la red es una combinación ***no lineal*** de las salidas de las neuronas de la capa oculta. Utilizaremos la ***función sigmoide***.

Para una entrada  

$$x = 
   \begin{pmatrix} 
      x_{11} \\ 
      x_{21} 
   \end{pmatrix}
$$

, el resultado de la capa de salida se calcula como:

$$
   y=\sigma(W_{e}x+b_{e})
$$

Donde la ***función sigmoide*** se define como:  

$$ \sigma(z) = 
   \frac{1}{1 + e^{-z}} 
$$

Donde 

$$
   z_ = 
      we_{11} \cdot x_{11} + we_{12} \cdot x_{21} + be
$$

La función sigmoide la podemos resolver de 3 modos:

1. Con una calculadora científica (puede ser la de Windows, por ejemplo)
2. [Con una calculadora en Internet](https://www.mathcelebrity.com/sigmoid-function-calculator.php)
3. [Verificando la y, para una x determinada sobre una gráfica de la función](https://www.desmos.com/calculator/84yf6yql2l?lang=es)

## Ejemplo de configuración de pesos y sesgos

Para que la red neuronal represente la función ***XOR***, podemos ajustar los pesos y los sesgos de la siguiente manera:

- $wh_{11} =  10, wh_{12} = -10, bh_{11} = -5$
- $wh_{21} = -10, wh_{22} =  10, bh_{12} = -5$
- $we_{11} =  15, we_{12} =  15, be      = -10$

***Nota***: Si la salida no coincidiese con la tabla de verdad de la puerta lógica XOR, deberíamos iterar pesos y sesgos (probar con otros diferentes, emulando un aprendizaje). Al final, la mayor parte de las iteraciones darían el resultado correcto y eso sería reflejado en las métricas.  

## Cálculo

1. **Entrada (0, 0)**:
   $$
      \begin{align}
      x_{11} &= 0;\\
      x_{21} &= 0;\\
      h_{1} &= \\
         &=ReLU(wh_{11}\cdot x_{11}+wh_{12}\cdot x_{21}+ bh_{1})\\
         &=ReLU(10\cdot0-10\cdot 0-5)\\
         &=ReLU(-5)\\
         & \approx 0\\
      h_{2} &= \\
         &=ReLU(wh_{21}\cdot x_{11} + wh_{22}\cdot x_{21} + bh_{2})\\
         &=ReLU(-10\cdot0+10\cdot0-5)\\
         &=ReLU(-5)\\
         & \approx 0\\
      y &= \\ 
         &=\sigma(15 \cdot 0 + 15 \cdot 0 + -10)\\
         &=\sigma(-10)\\
         & \approx 0\\
      \end{align}
   $$

2. **Entrada (0, 1)**:
   $$
      \begin{align}
      x_{11} &= 0;\\
      x_{21} &= 1;\\
      h_{1} &= \\
         &=ReLU(wh_{11}\cdot x_{11}+wh_{12}\cdot x_{21}+ bh_{1})\\
         &=ReLU(10\cdot 0-10\cdot 1-5)\\
         &=ReLU(-15)\\
         & \approx 0\\
      h_{2} &= \\
         &=ReLU(wh_{21}\cdot x_{11} + wh_{22}\cdot x_{21} + bh_{2})\\
         &=ReLU(-10\cdot0+10\cdot1-5)\\
         &=ReLU(5)\\
         &=5\\
      y &= \\ 
         &=\sigma(15\cdot 0+15\cdot 5-10)\\
         &=\sigma(65)\\
         &= 1\\
      \end{align}
   $$

3. **Entrada (1, 0)**:
   $$
      \begin{align}
      x_{11} &= 0;\\
      x_{21} &= 1;\\
      h_{1} &= \\
         &=ReLU(wh_{11}\cdot x_{11}+wh_{12}\cdot x_{21}+ bh_{1})\\
         &=ReLU(10\cdot 1-10 \cdot 0-5)\\
         &=ReLU(5)\\
         &=5\\
      h_{2} &= \\
         &=ReLU(wh_{21}\cdot x_{11} + wh_{22}\cdot x_{21} + bh_{2})\\
         &=ReLU(-10\cdot 1+10\cdot 0-5)\\
         &=ReLU(-15)\\
         &=0\\
      y &= \\ 
         &=\sigma(15\cdot 5+15\cdot 0-10)\\
         &=\sigma(65)\\
         &= 1\\
      \end{align}
   $$

4. **Entrada (1, 1)**:
   $$
      \begin{align}
      x_{11} &= 0;\\
      x_{21} &= 1;\\
      h_{1} &= \\
         &=ReLU(wh_{11}\cdot x_{11}+wh_{12}\cdot x_{21}+ bh_{1})\\
         &=ReLU(10\cdot 1-10\cdot 1-5)\\
         &=ReLU(-5)\\
         &=0\\
      h_{2} &= \\
         &=ReLU(wh_{21}\cdot x_{11} + wh_{22}\cdot x_{21} + bh_{2})\\
         &==ReLU(-10\cdot 1+10\cdot 1-5)\\
         &=ReLU(-15)\\
         &=0\\
      y &= \\ 
         &==\sigma(15\cdot 0+15\cdot 0-10)\\
         &=\sigma(-10)\\
         &\approx 0\\
      \end{align}
   $$

Comparando con la tabla de verdad de la puerta XOR, comprobamos que ***las salidas de la RNA son correctos***.  

|x_{11}|x_{21}|x_{11} RNA(XOR) x_{21}|
|-|-|:-:|
|1|0|1|
|0|1|1|
|0|0|0|
|1|1|0|

## Conclusión

Como se puede ver, con esta configuración de pesos y sesgos, la red neuronal puede aproximar la función ***XOR*** utilizando la ***función de activación ReLU*** en la capa oculta y la ***función de activación sigmoide*** en la capa de salida, con dos neuronas de entrada y ***dos en la capa oculta***.
