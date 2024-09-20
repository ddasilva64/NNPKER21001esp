# Definición del problema


## El cliente

***Adventure Works Cycles, Inc.*** tiene el almacén central de distribución para el estado de Colorado en Commerce City (Condado de Adams), situado a 11,1 Km de la capital del estado (Denver).

Tiene una plantilla de almacén formada por 37 operarios. Todos trabajan por igual en dos turnos de 8 h (uno de día y otro de noche), en cada turno, dos de los operarios son encargados. Dos empleados, son también encargados de mantenimiento. Además, el cielnte, tiene una plantilla de 26 empleados administrativos directamente implicados en la gestión del almacén. El personal, en general, muestra muchas ineficiencias y en redes sociales las reclamaciones por mal servicio suponen un 98% del total. Nadie atiende las reclamaciones de los clientes de manera sistemática, ni hay el mínimo interés en el staff directivo por el clima laboral.

La empresa dispone de un SGA (Sistema de Gestión de Almacén), que ha ido creciendo (de forma incontrolada) con los años y que ahora opera en áreas que no son estríctamente su ámbito, es decir, el gran sistema fagocita subsistemas que no le son propios. Además, sin conexión ni interface, dispone de un software de Contabilidad y otro de gestión de nóminas.

Además, se supervisan manualmente las transacciones enviadas desde y hacia las oficinas de la compañía en la ciudad de Denver. El soft que utiliza la central de Denver es un conocido soft europeo, caro y complejo.

## El problema

La disminución en las ventas de los últimos años exigen revinversiones y diversificaciones que implican un plan de optimización de recursos. Para conseguir este propósito se necesita incrementar la productividad de almacén en Commerce City que permita cerrar otros almacenes logísticos más pequeños y menos productivos en los estados adyacentes de Wyoming, Nuevo México, Kansas y Arizona.

En una fase futura se quiere extender la cadena logística de los clientes hasta llegar a anticipar sus necesidades y hacer propuestas de pedidos. 

De los 26 empleados administrativos directamente implicados en la gestión del almacén, se quiere reducir la plantilla a 7 empleados y un manager de operaciones (sin encargados).

## La solución 

La gestión de la operación del almacén se divide en tres partes:  

1. Corto plazo: gestión del dia a dia
2. Medio plazo: Gestión a 6 meses o 1 año de las operaciones (táctica)
3. Largo plazo: Gestión 1 año o 5 años de las operaciones (estratégica)

En una primera fase se decide abordar el punto 1, es decir, la gestión del dia a dia. Para ello, se hará una gestión en 3 tiempos:

1. Antes: Previsión de jornada laboral
2. Durante: Control de la jornada laboral
3. Después: Evaluación de la jornada laboral

Para realizar las tareas anteriores se decide utilizar técnicas de ML, las cuales serán desarrolladas por nuestra empresa (***3DoWoCo***). El proyecto con el que se implementarán las soluciones, será conocido como ***WEMPRO*** project (**W**arehouse **E**mployee **M**anagement **PRO**ject).

### El detalle (secciones)

Control de la productividad de los operarios de almacén, con IA, antes, durante y después de la jornada laboral:

a) ***Antes de la Jornada Laboral (planificación de tareas)***: Aplicaremos modelos de Aprendizaje Supervisado, para poder predecir la carga de trabajo diaria basándose en datos históricos, como el volumen de pedidos y la disponibilidad de personal. Los algoritmos pueden ser ***Decision Trees, Random Forests*** y ***RNAs***. Además utilizaremos técnicas de clustering, como ***K-means***, para poder ***agrupar tareas similares y asignarlas a los empleados más adecuados, optimizando así la distribución del trabajo***.

- ***Optimización de rutas dentro del almacén***
    - Algoritmos de ruta: Algoritmos como Dijkstra o A\* para encontrar las rutas más cortas y eficientes dentro del almacén.
    - Algoritmos Genéticos: Estos algoritmos pueden encontrar soluciones óptimas para problemas complejos de rutas, simulando el proceso de evolución natural.
    - Redes Neuronales Artificiales (RNA): Las RNA pueden aprender de datos históricos y en tiempo real para mejorar continuamente las rutas dentro del almacén. Por ejemplo, una red neuronal convolucional (CNN) puede analizar imágenes del almacén para identificar obstáculos y optimizar las rutas.

- ***Análisis predictivo de picos de demanda***
    - ***Modelos predictivos***: Utilizar técnicas de ***regresión lineal*** y de análisis de series temporales (como ARIMA) para predecir picos de demanda, basándose en datos históricos.
    - Redes Neuronales Recurrentes (RNN): Para analizar patrones en datos secuenciales, en datos históricos y predecir futuras necesidades, especialmente las LSTM (Long Short-Term Memory).

- ***Formación personalizada de los operarios***
    - ***Sistemas de recomendación***: Utilizar técnicas de ***filtrado colaborativo y contenido basado en el aprendizaje***, para recomendar cursos de formación específicos a cada empleado, basándose en sus necesidades y rendimiento pasado.
    - ***Aprendizaje adaptativo***: ***Plataformas que ajustan el contenido de formación en función del progreso y las necesidades del empleado***, utilizando algoritmos de aprendizaje supervisado y no supervisado.

b) Durante la Jornada Laboral (monitoreo en tiempo real): Utilizaremos algoritmos de visión por computador, como las CNN, los cuales pueden monitorear el rendimiento y la seguridad en tiempo real, analizando imágenes y videos del almacén. Además realizaremos un Análisis de Datos en Tiempo Real, con herramientas de análisis de flujo de datos, los cuales pueden utilizar modelos de Machine Learning para detectar anomalías y optimizar procesos en tiempo real.

- Asistentes virtuales para soporte inmediato a los operarios
    - Chatbots y asistentes virtuales: Utilizar Procesamiento de Lenguaje Natural o NLP (modelos como BERT o GPT), para crear asistentes que puedan responder preguntas y proporcionar soporte en tiempo real, mejorando la eficiencia y reduciendo el tiempo de inactividad.

- Análisis de datos en tiempo real para identificar cuellos de botella y áreas de mejora
    - Big Data Analytics: Utilizaremos técnicas de machine learning, como clustering y clasificación, para el análisis de grandes volúmenes de datos para identificar patrones y tendencias, que mejoren la eficiencia del almacén.
    - Modelos Predictivos: Utilizaremos modelos de Machine Learning, que pueden predecir problemas potenciales y optimizar procesos basándose en datos históricos y en tiempo real, para optimizar procesos y mejorar la eficiencia.  

c) ***Después de la Jornada Laboral (evaluación del rendimiento)***: Haremos Análisis de Sentimiento (utilizando técnicas de NLP), para analizar comentarios y retroalimentación de los operarios, para evaluar su satisfacción y rendimiento. Además, ***utlizaremos algoritmos que evalúan el rendimiento de los operarios basándose en métricas predefinidas, proporcionando retroalimentación detallada y personalizada***.

- ***Informes automatizados que resuman la productividad del día y destaquen áreas de mejora***.
    - Generación de Lenguaje Natural (NLG): Utilizaremos herramientas que convierten datos en informes escritos automáticamente, utilizando modelos de machine learning para generar texto coherente y relevante.
    - ***Dashboards interactivos***: Utilizaremos ***herramientas de visualización de datos y Machine Learning, para crear informes interactivos***, que resuman la productividad del día y destaquen áreas de mejora.

- Mantenimiento Predictivo, para predecir y planificar el mantenimiento de equipos y maquinaria, asegurando que todo esté en óptimas condiciones para el siguiente día de trabajo.
    - Modelos predictivos: Utilizaremos algoritmos de Machine Learning, como los modelos de regresión y las redes neuronales, para predecir fallos en equipos y planificar el mantenimiento, de manera proactiva.
    - IoT y sensores: Utilizaremos sensores conectados a sistemas de IA para monitorear el estado de los equipos en tiempo real, utilizando Machine Learning para analizar los datos y predecir problemas antes de que ocurran.

- ***Análisis de tendencias a largo plazo en la productividad y hacer ajustes estratégicos en la gestión del almacén***.
    - ***Análisis de Series Temporales***: Utilizaremos técnicas de ***análisis de series temporales y Machine Learning, para identificar tendencias a largo plazo***, en la productividad y hacer ajustes estratégicos en la gestión del almacén.
    - Redes Neuronales: Utilizaremos modelos avanzados de redes neuronales, para analizar grandes volúmenes de datos y detectar patrones, que no son evidentes a simple vista.

### Fase 01

La tareas abordadas en el proyecto ***WEMPRO01*** serán las siguientes:  

1. En la sección A:  

    1.1. Utilizaremos el algoritmo ***K-means***, ***para obtener*** clases de actividades (***categorías***). 
    1.2. Una vez identificadas las ***categorías***, ***etiquetaremos*** a los operarios por la misma. De otro modo, mezclaríamos categorías distintas y no obtendríamos resultados válidos.
    1.3. Una vez etiquetados los operarios por su ***categoría***, ***clasificaremos*** (con ***Decision Trees*** y/o ***Random Forest***) la actividad ***por tipos de productividad*** (de menor a mayor) y ***asignaremos el tipo a cada operario***.
    1.4. La ***plantilla ideal*** la obtendremos ***cruzando datos de categorías*** (clases de actividad) y ***tipo de operario*** (de más productivos a menos productivos).
    1.5. Se hará un estudio del perfil completo de cada empleado, cruzado con su posición en el ranking de la plantilla ideal, mediante ***Decision Trees*** y/o ***Random Forest***, para conocer el ***riesgo de burnout en los buenos operarios y de facilidad de despido en los malos***. Este iforme será confidencial y lo gestionará RRHH.
    1.6. Generaremos Informes en ***Power BI*** con implementación de ***RLS*** (Row Level Security), para ***establecer la información que deba ser visible para cada nivel del staff de la empresa***.
    1.7. Guardaremos un histórico de la planificación.

2. En la sección C:  

    2.1. Mostraremos información histórica hasta la fecha de la jornada, en informes de ***Power BI***.
    2.2. Mostraremos una comparación de lo planificado con lo efectuado en ***Power BI***.
