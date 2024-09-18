# Notas de Análisis matemático

<details>

<summary>Índice de contenidos</summary>

* [***Función sigmoide***](#función-sigmoide)
    * [Regiones de saturación]()


</details>

## Función sigmoide

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Función sigmoide</span>][r000]***: A.k.a. (also known as) ***función logística***, es la función matemática que aparece en diversos modelos de crecimiento de poblaciones, propagación de enfermedades epidémicas, difusión en redes sociales y como función de activación en RNA (Redes Neuronales Artificiales). Dicha función constituye un refinamiento del modelo exponencial para el crecimiento de una magnitud. Modela la función sigmoidea de crecimiento de un conjunto P.

$$
    f(x) = \frac{1}{(1 + e^{-x})}
$$

![Función sigmoide][i000] 

Se transforma fácilmente en la función logística

$$
S (x) = {\frac{1}{1+e^{- x}}}={\frac{e^{x}}{e^{x}+1}}=1-S(-x)
$$

### Características

- ***Trascendente***

    Es trascendente, porque es exponencial.

- ***Dominio***  

    $\operatorname {Dom} (f)=\mathbb{R}-\lbrace 0 \rbrace$. Es exponencial y fraccionaria.

- ***Asíntotas***  

    - Verticales 

        No

    - Horizontales

        $\lim _{x\to -\infty}{\frac{1}{1+e^{-x}}}=\frac{1}{-\infty}=0$ (horizontal); $y=0$  
        
        $\lim _{x\to +\infty}{\frac{1}{1+e^{-x}}}=\frac{1}{+\infty}=1$ (horizontal); $y=1$

    - Oblícuas

- Primera derivada
    
    $y'=\cfrac{dy}{dx}=\frac{e^{-x}}{(1+e^{-x})^{2}}$  
    
    Derivada simple:
    $y'=y(1-y)$

- Concavidad

- Convexidad

- Puntos de inflexión

    Tiene un punto de inflexión.
    $x=0$

- Crecimiento
    Creciente (con parámetros decreciente)

    Crecimiento acotado, para todo t se cumple que: $0<M(t)<K$
 
    Aproximación exponencial, para valores pequeños de M/K (o también para valores de $t\to -\infty$ la función logística puede aproximarse por un modelo de crecimiento continuo del tipo $M(t)\approx M'e^{rt}$.  

    Valores límite, la función logística generalizada dada por la ecuación (2) tiene los siguientes límites: $\lim _{t\to -\infty }P(t)=a{\frac {m}{n}},\quad P(0)=a{\frac {1+m}{1+n}},\quad \lim _{t\to \infty }P(t)=a$

### Regiones de saturación

Esta función de activación toma cualquier rango de valores a la entrada y los mapea al rango de 0 a 1 a la salida. Dicho comportamiento se muestra en la siguiente figura:

![Regiones de saturación de la función sigmoide][i000a] 

En la actualidad esta función de activación tiene un uso limitado, y realmente su principal aplicación es la clasificación binaria.

Lo anterior se debe al problema de saturación. Como se observa en la figura anterior, la función se satura a 1 cuando la entrada (z) es muy alta, y a 0 cuando es muy baja. Esto hace que durante el entrenamiento usando el método del gradiente descendente, los gradientes calculados sean muy pequeños, dificultando así la convergencia del algoritmo.

## Listado de referencias externas

* Función sigmoide (Wikipedia)  

[r000]: https://es.wikipedia.org/wiki/Funci%C3%B3n_sigmoide "referencia a función sigmoide en Wikipedia"



## Listado de imágenes

* Función sigmoide  

[i000]: https://i.imgur.com/XhEF5Rf.png "Función sigmoide"

* Regiones de saturación de la función sigmoide  

[i000a]: https://i.imgur.com/8wkIfHG.png "Regiones de saturación de la función sigmoide"




