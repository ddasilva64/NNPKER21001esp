# Notas de Álgebra lineal

<details>

<summary>Índice de contenidos</summary>

* [***Funciones lineales***](#funciones-lineales)
    * [***Características***](#características)
    * [***Funciones lineales de diversas variables***](#funciones-lineales-de-diversas-variables)
* [***Funciones no lineales***](#funciones-no-lineales)
    * [***Tipos***](#tipos)
* [***Hipersuperfície***](#hipersuperfície)
* [***Hiperplano***](#hiperplano)
* [***Espacio topológico***](#espacio-topológico)
* [***Homeomorfismo***](#homeomorfismo)
* [***Espacio euclídeo***](#espacio-euclídeo)
* [***Variedad topológica***](#variedad-topológica)

</details>

## Funciones lineales

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Funciones lineales</span>][r000]***: Una función lineal se representa en el plano cartesiano como la gráfica de una recta.  

### Características

Una función lineal es una ***función polinómica de primer grado***, es decir, una función de una variable (normalmente esta variable se denota con x), que ***puede ser escrita como la suma de términos de la forma $ax^n$*** (donde ***a es un número real y n es un número natural***) donde $n\in\{0, 1\}$; es decir, n solo puede ser 0 o 1. 

$f(x)= mx+b$

donde  
$\{m, b, x\}\in \mathbb{R};$  
m es la pendiente (inclinación) de la recta;  
b es el punto de corte de la recta con el eje vertical (y)  

![Funciones lineales][i000]  

### Funciones lineales de diversas variables

Las funciones lineales de diversas variables admiten también interpretaciones geométricas. Así una función lineal de dos variables de la forma

$f(x,y)=a_{1}x+a_{2}y$  
Representa un plano y una función
$f(x_{1},x_{2},\dots,x_{n})=a_{1}x_{1}+a_{2}x_{2}+\dots +a_{n}x_{n}$  
Representa una hipersuperficie plana de dimensión n y pasa por el origen de coordenadas en un espacio (n + 1)-dimensional.  

## Funciones no lineales

***[<span style="font-family:Verdana; font-size:0.95em;color:red">Funciones no lineales</span>][r001]***: Una función no lineal se representa en el plano cartesiano como una gráfica distinta de una recta (parábolas, curvas cúbicas, hipérboles, etc.). Por lo tanto, ***las funciones polinómicas de primer grado son las únicas que no son funciones no lineales***.

![Funciones no lineales][i001] 

### Tipos

1. ***Funciones cuadráticas***  

Una función cuadrática es una función polinómica de segundo grado. Por lo tanto, la fórmula de una función cuadrática es la siguiente:  

$f(x)=ax^{2}+bx+c$

Donde $ax^{2}$ es el término cuadrático, $bx$ el término lineal y $c$ el término independiente de la función polinómica.

Ejemplos de funciones cuadráticas o funciones polinómicas de segundo grado:

$f(x)=3x^{2}+5x+1$
$f(x)=-7x^{2}+3x+4$

La gráfica de una función cuadrática es siempre es una parábola. Sin embargo, la forma de la parábola depende del signo del coeficiente principal a de la función. 

2. ***Funciones de proporcionalidad inversa***  Las funciones exponenciales son aquellas funciones no lineales en las que la variable independiente x aparece en el exponente de una potencia. Es decir, una función exponencial es de la siguiente forma:

f(x)=a^x

Donde a es un número real positivo y diferente de 1.

Tal y como su propio nombre indica, la gráfica de una función exponencial crece exponencialmente, así que hay que calcular más puntos de la función para representarla correctamente.

Una función de proporcionalidad inversa es aquella función que relaciona dos magnitudes que son inversamente proporcionales. Es decir, una aumenta cuando la otra disminuye y viceversa.  

Este tipo de funciones no lineales vienen definidas por la siguiente fórmula:

$y=\frac{k}{x}$

Donde k es una constante llamada razón de proporcionalidad.

Ejemplos de funciones de proporcionalidad inversa:

$y=\frac{5}{x}$  
$y=\frac{-4}{x}$  
$y=\frac{2}{x+1}$

Las funciones de proporcionalidad inversa siempre tienen asíntotas.  

3. ***Funciones irracionales***  

Una función irracional, también llamada función radical, es una función no lineal que tiene la variable independiente x bajo el símbolo de una raíz.

El resultado de una raíz puede ser positivo o negativo. De manera que la representación de una función irracional (o radical) tiene dos posibles curvas, aunque normalmente se es representa solo la rama positiva.

4. ***Funciones exponenciales***  

Las funciones exponenciales son aquellas funciones no lineales en las que la variable independiente x aparece en el exponente de una potencia. Es decir, una función exponencial es de la siguiente forma:

$f(x)=a^{x}$

Donde a es un número real positivo y diferente de 1.

Tal y como su propio nombre indica, la gráfica de una función exponencial crece exponencialmente.

5. ***Funciones logarítmicas***

Las funciones logarítmicas son aquellas funciones cuya variable independiente x forma parte del argumento de un logaritmo. Es decir, una función logarítmica es una función no lineal que presenta la siguiente forma:

$f(x)=\log_a x$

Donde a es obligatoriamente un número real positivo y diferente de 1.

La inversa de la función logarítmica es la función exponencial. De modo que las gráficas de una función logarítmica y una función exponencial son simétricas respecto de la recta y=x si ambas poseen la misma base.


## Hipersuperfície

En Matemáticas, una hipersuperficie es una variedad n-dimensional con n > 2, es decir, un objeto geométrico que generaliza la noción de una superficie bidimensional a dimensiones superiores, del mismo modo que el ***hiperplano*** generaliza la noción de plano.

Técnicamente, una hipersuperficie de dimensión n es un espacio topológico que es localmente homeomorfo al espacio euclídeo $\mathbb{R} ^{n}$. Ello significa que para cada punto P de la hipersuperficie hay una vecindad de P (una pequeña región que la rodea) que es homeomorfa a un disco abierto de 
$\mathbb{R}^{n}$. Eso permite definir una serie de coordenadas locales que parametrizan dicha hipersuperficie.

El tipo más simple de hipersuperficie son las 3-variedades contenidas en el espacio de cuatro dimensiones $\mathbb{R}^{4}$.  

![Hipersuperfície][i002] 

## Hiperplano

En Geometría, un hiperplano es una extensión del concepto de plano.

En un espacio unidimensional (como una recta), un hiperplano es un punto: divide una línea en dos líneas. En un espacio bidimensional (como el plano xy), un hiperplano es una recta: divide el plano en dos mitades. En un espacio tridimensional, un hiperplano es un plano corriente: divide el espacio en dos mitades. Este concepto también puede ser aplicado a espacios de cuatro dimensiones y más, donde estos objetos divisores se llaman simplemente hiperplanos, ya que la finalidad de esta nomenclatura es la de relacionar la geometría con el plano.   

![Hiperplano][i003] 

## Espacio topológico

Un espacio topológico es una estructura matemática que permite la definición formal de conceptos como convergencia, conectividad, continuidad y vecindad, usando subconjuntos de un conjunto dado.1​ La rama de las matemáticas que estudia los espacios topológicos se llama topología. Las variedades, al igual que los espacios métricos, son especializaciones de espacios topológicos con restricciones y estructuras propias.  

![Espacio topológico][i004] 

## Homeomorfismo

De modo intuitivo, el concepto de homeomorfismo refleja cómo dos espacios topológicos son "los mismos" vistos de otra manera: permitiendo estirar, doblar o cortar y pegar. Sin embargo, los criterios intuitivos de "estirar", "doblar", "cortar y pegar" requieren de cierta práctica para aplicarlos correctamente. Deformar un segmento de línea hasta un punto no está permitido, por ejemplo. Contraer de manera continua un intervalo hasta un punto es otro proceso topológico de deformación llamado homotopía.

![Homeomorfismo][i005] 

## Espacio euclídeo

El espacio euclídeo (también llamado espacio euclidiano) es un tipo de espacio geométrico donde se satisfacen los axiomas de Euclides de la geometría. La recta real, el plano euclídeo y el espacio tridimensional de la geometría euclidiana son casos especiales de espacios euclidianos de dimensiones 1, 2 y 3 respectivamente. El concepto como conjunto, Rn es la serie de n-adas ordenadas de números reales. Es decir:

$\mathbb{R}^{n}= \{=x_{1}+x_{2}+\dots +x_{n}\}|x_{i}\in\mathbb{R}, i= 1,2,…,n\}$

caracterizamos un elemento $\mathbb{R}^{n}$ por $x=(x_{1}+x_{2}+\dots +x_{n})$. Denotados dos elementos $x,y\in\mathbb{R}^{n}$ podemos decir que $x=y\iff\ x_{i}=y_{i}\ \forall\ i=1,2,…,n$  

![Espacio euclídeo][i006] 

## Variedad topológica

En Matemática, una variedad es el objeto geométrico estándar que generaliza la noción intuitiva de "curva" (1-variedad) y de "superficie" (2-variedad) a cualquier dimensión y sobre cuerpos diversos (no solamente el de los reales, sino también complejos y matriciales).

![Variedad topológica][i007] 

## Listado de referencias externas

* Funciones lineales (Wikipedia)  

[r000]: https://es.wikipedia.org/wiki/Funci%C3%B3n_lineal "referencia a funciones lineales en Wikipedia"

* Funciones no lineales (Wikipedia)  

[r001]: https://www.funciones.xyz/funciones-no-lineales/ "referencia a funciones no lineales en Wikipedia"

* Hipersuperfície (Wikipedia)  

[r002]: https://es.wikipedia.org/wiki/Hipersuperficie "referencia a hipersuperfície en Wikipedia"

* Hiperplano (Wikipedia)  

[r003]: https://es.wikipedia.org/wiki/Hiperplano "referencia a hiperplano en Wikipedia"

* Espacio topológico (Wikipedia)  

[r004]: https://es.wikipedia.org/wiki/Espacio_topol%C3%B3gico "referencia a espacio topológico en Wikipedia"

* Homeomorfismo (Wikipedia)  

[r005]: https://es.wikipedia.org/wiki/Homeomorfismo "referencia a homeomorfismo en Wikipedia"

* Espacio euclídeo (Wikipedia)  

[r006]: https://es.wikipedia.org/wiki/Espacio_eucl%C3%ADdeo "referencia a espacio euclídeo en Wikipedia"


## Listado de imágenes

* Funciones lineales  

[i000]: https://i.imgur.com/qR4YP8f.png "Funciones lineales"

* Funciones no lineales  

[i001]: https://i.imgur.com/RXxZOh2.png "Funciones no lineales"

* Hipersuperfície  

[i002]: https://i.imgur.com/rXePqUg.png "Hipersuperfície"

* Hiperplano  

[i003]: https://i.imgur.com/h94OoEV.png "Hiperplano"

* Espacio topológico  

[i004]: https://i.imgur.com/LDaZTQK.png "Espacio topológico"

* Homeomorfismo  

[i005]: https://i.imgur.com/SdAEOir.gifv "Homeomorfismo"

* Espacio euclídeo  

[i006]: https://i.imgur.com/AtCTSQs.png "Espacio euclídeo"