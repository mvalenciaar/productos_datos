# **Trabajo Final Productos de Datos**

### **Estudiantes:**
- David Arango Estrada - email: darangoe@unal.edu.co
- Laura Ávila Roa - email: laavilaro@unal.edu.co
- Juan Pablo Arcila - email: jarcilal@unal.edu.co
- Alejandro Montoya Restrepo - email: amontoyar@unal.edu.co
- Maria Victoria Valencia Arango - email: mvalenciaar@unal.edu.co
- Juan Fernando Yanes Doria - email: jyanes@unal.edu.co

### **Tema: Fraude en tarjetas de crédito**

#### **Introducción**
El fraude con tarjeta de crédito implica el uso no autorizado de la información de la 
tarjeta de crédito de una persona con el propósito de cargar compras en la cuenta de la 
víctima o extraer fondos de su cuenta [1]. Es considerado como una forma de robo.

El panorama global indica que las compras físicas o por sucursales ha disminuido y el 
volumen de pagos y compras por internet aumenta considerablemente. Adicionalmente, 
respondiendo a las brechas de seguridad en los actuales sistemas de información que 
pueden acarrear costos elevados para las compañías. Según un análisis de Datos de 
Nilson Report, las pérdidas mundiales por fraude con tarjetas de crédito se elevaron a 
más de $21000 millones de dólares en 2015, en comparación con 8000 millones de 
dólares registrados en 2010 y para el 2020, se proyectan pérdidas de $31.000 millones 
de dólares [2].

### **Comprensión del negocio**
Las organizaciones bancarias, se ven sujetas a la regulación de sus actividades 
financieras lo que implica el control de las acciones de captación de dinero. Teniendo 
en cuenta este panorama, el estado establece medidas y canales de seguridad que les 
permite a los consumidores la confianza y adaptación a los medios electrónicos para 
propiciar la transformación de los servicios financieros en la era digital. En este sentido, 
a medida que aumentan las transacciones digitales, el riesgo de ciertos delitos, por 
ejemplo, la apropiación de los activos digitales del cliente, aumentan. Es por esto que 
las compañías vienen adelantando campañas para la prevención de este tipo de fraude, 
asimismo incentivando el uso de tecnologías como el Machine Learning y la minería de 
datos para marcar la diferencia en cuanto a la efectividad de la detección y la prevención
de delitos, mediante procesos y modelos que ayudan en la detección de transacciones 
fraudulentas, permitiendo construir ecosistemas bancarios más seguros [3] [4].

### **Definición del problema**


La entidad bancaria ha presentado distintos casos de fraude en sus clientes con el uso de tarjeta de crédito, lo que ha impactado de manera negativa su reputación. De modo que, la presidencia toma la decisión de evaluar por georreferenciación las transacciones que presentan un comportamiento fraudulento. Dichas transacciones son capturadas para que posteriormente Camilo ingeniero de datos del área de fraude, se encargue de realizar las validaciones pertinentes con el fin de identificar lo sucedido con las transacciones fraudulentas. Por otra parte, José del área de ciberseguridad será el encargado de discriminar los puntos de georreferenciación y finalmente Carolina gerente de riesgos y fraudes exhibirá los resultados a la presidencia de la empresa.

La entidad cuenta con los siguientes sistemas para capturar la información:
- La captura transaccional por medio de AS400
- La captura de coordenadas de georeferenciación por un proveedor externo.

La propuesta se entregará a presidencia, posteriormente se desarrollará el piloto para poner en producción el desarrollo y así disminuir la materialización de fraudes por tarjeta de crédito.


### **Flujo de trabajo**


![work_flow_update_final](https://user-images.githubusercontent.com/56141354/220983415-2d0454a3-92ef-4632-b093-7bc30c04702a.jpeg)



- Compras realizadas por tarjetahabientes.
- Proceso de validación de la transacción.
- Estado final: Transacción fraudulenta o no

### **Pregunta de negocio:** 

¿Cómo disminuir el riesgo reputacional por el impacto de las transacciones fraudulentas generadas por compras realizadas con TC en comercio físico y online?

### **Preguntas analíticas:**

¿Cuál es la probabilidad de que una compra realizada con tarjeta de crédito sea fraudulenta?

### **Tabla del sistema transaccional**

La tabla del sistema transaccional de las compras realizadas con tarjeta de crédito contiene los siguientes campos:

* `distance_from_home`: Campo de distancia desde casa.
* `distance_from_last_transaction`: Campo de distancia desde la última transacción.
* `ratio_to_median_purchase_price`: Relación con el precio promedio de compra.
* `repeat_retailer`: Comercio habitual.
* `used_chip`: Si se usa chip.
* `used_pin_number`: Si se usa número de PIN.
* `online_order`: Pedido en línea.
* `fraud`: Si existe fraude.

### **Simulador**

El código de simulación se encuentra contenido dentro de la carpeta src-preparation, como se específica a continuación:
- purchases_cardholder.py
- onsite_valid.py
- onsite_state.py
- online_valid.py
- online_state.py

### **Entrega 1: Procesamiento de datos**

El código desarrollado para este caso de uso, presenta la siguiente estrucura definida para la construcción del pipeline final para cada entrega:

- Carpeta src: Es la carpeta donde estará el código fuente y el archivo de requirements.txt donde se encuentran las librerías necesarias de ejecución de código.
    - subcarpeta: preparation:
      - card_transdata.csv: dataset de trabajo sobre transacciones realizadas por compras con tarjetas de crédito en comercios onsite y online, así como, el resultado de si se presentó fraude o no (archivo card_transdata.csv).
      - Makefile: Orquestador de la simulación.
      - preparation.py: Allí se encuentran las funciones definidaspara los procesos de cargue, limpieza, y simulaciones, esta información queda almacenada en el repositorio local, y en nube.
      - Archivos para la silmulación: Estas se plantean de acuerdo al flujo de trabajo propuesto.
        - purchases_cardholder.py: Hace referencia a la simulación sobre la cantidad de compras realizadas por medio de tarjeta de crédito por parte de los tarjetahabientes.
        - onsite_valid.py: Hace referencia a la simulación sobre el proceso de validación de las transacciones realizadas por medio de tarjeta de crédito en comercio físico.
        - onsite_state.py: Hace referencia a la simulación sobre el estado final de la transacción de compra realizada por medio de tarjeta de crédito en comercio físico, arrojando si corresponde a un fraude o no.
        - online_valid.py: Hace referencia a la simulación sobre el proceso de validación de las transacciones realizadas por medio de tarjeta de crédito en comercio online.
        - online_state.py: Hace referencia a la simulación sobre el estado final de la transacción de compra realizada por medio de tarjeta de crédito en comercio online.

### **Entrega 1: Ejecución y flujo de la simulación**

**Se recomienda ejecutar la simulación estando ubicados en la carpera 'src' del proyecto**

En la simulación se comprenden las 3 etapas descritas en el fujo de trabajo.

-Para comenzar la simulación se deben tener compras en el sistema, para esto podemos escoger si queremos tener todas las compras del dataset disponibles, o un número determinado de compras para evaluarlas.
Para comprar entonces se tienen 2 opciones del comando `purchases_cardholder`:

  1. Si se quieren hacer todas las compras del dataset, se ejecuta el comando `make -C preparation/ purchases_cardholder`
     
     El archivo de orquestación 'Makefile' identifica que se quieren comprar todas las filas del dataset ya que no posee parámetro el llamado a la función, de esta manera procede a invocar al archivo 'purchases_cardholder.py' el cual a su vez verifica la cantidad de argumentos con la que fué llamado, como solo es 1 (purchases_cardholder) procede a buscar en el archivo 'preparation.py' la función correspondiente a 'purchases_cardholder' con argumento por defecto.
    En esta función se llama a las funciones de carga (load_file_card) y limpieza de datos (cleansing_data) y posterior a ejecutarlas procede a 'comprar'.
    Al comprar lee la cantidad de registros pedida (la totalidad del dataset en este caso) y devuelve el mensaje "La cantidad de registros de compras es 1000000".
    
   Imagen de ejecución:
   
   ![image](https://user-images.githubusercontent.com/17460738/221286362-d387b6f7-d0f4-474f-8c25-93c9d58c0a69.png)    
   
   De esta manera se contará con las compras completas del dataset.
    
  2. Si se quieren comprar n cantidad de registros, se ejecuta el comando `make -C preparation/ purchase_cardholder n=x`
     Donde 'x' se reemplaza por un número entero de compras que se desean realizar.
     
     El archivo de orquestación 'Makefile' identifica que se quieren comprar una cantidad fija de elementos, por lo que procede a invocar al archivo 'purchases_cardholder.py' el cual a su vez verifica la cantidad de argumentos con la que fué llamado, 2 en este caso (purchases_cardholder y n=20), procede a buscar en el archivo 'preparation.py' la función correspondiente a 'purchases_cardholder(n)', con paso de argumento.
    En esta función se llama a las funciones de carga (load_file_card) y limpieza de datos (cleansing_data) y posterior a ejecutarlas procede a 'comprar'.
    Al comprar lee la cantidad de registros pedida (el x pasado en la consola) y devuelve el mensaje "La cantidad de registros de compras es n".
    
   Imagen de ejecucion con 20 compras:
    
   ![image](https://user-images.githubusercontent.com/17460738/221287770-29b2b01b-638e-46b7-bae2-e2317a8acf54.png)
    
   De esta manera tendremos n compras del dataset disponibles para análisis

- Luego de haber realizado las compras, sigue la etapa de verificar el medio por el cual se realizaron las compras, segmentando las transacciones en compras Online y en compras Onsite.
  Para validar el medio de las comprar, se tienen entonces 2 comandos:
  
  1. Validar compras onsite mediante el comando `make -C preparation/ onsite_valid`
     
     El archivo de orquestación 'Makefile' identifica que se quieren validar de las compras realizadas, cuáles fueron onsite, para esto procede a invocar al archivo 'onsite_valid.py' el cual a su vez busca en el archivo 'preparation.py' la función correspondiente a 'onsite_transactions_validation'.
    En esta función se llama a las funciones de carga (load_file_card) y limpieza de datos (cleansing_data) y posterior a ejecutarlas procede a 'validar'.
    Al validar, filtra las compras que tienen el uso de chip = 1 y devuelve el mensaje "Se validan n transacciones Onsite".
    
    Imagen de ejecucion:
    
    ![image](https://user-images.githubusercontent.com/17460738/221296791-ec9e98a7-181f-4790-990a-ecf537c8ca0b.png)
    
    De esta manera se validan las compras Onsite.
    
  2. Validar compras online mediante el comando `make -C preparation/ online_valid`
     
     El archivo de orquestación 'Makefile' identifica que se quieren validar de las compras realizadas, cuáles fueron online, para esto procede a invocar al archivo 'online_valid.py' el cual a su vez busca en el archivo 'preparation.py' la función correspondiente a 'online_transactions_validation'.
    En esta función se llama a las funciones de carga (load_file_card) y limpieza de datos (cleansing_data) y posterior a ejecutarlas procede a 'validar'.
    Al validar, filtra las compras que tienen el atrubito compra online = 1 y devuelve el mensaje "Se validan n transacciones Online".
    
    Imagen de ejecucion:
    
    ![image](https://user-images.githubusercontent.com/17460738/221297667-0db03904-bcd7-4a75-a513-c3f5e6904905.png)
    
    De esta manera se validan las compras Online.
    
- Luego de haber validado las compras, se procede a la etapa de identificación de la cantidad de fraudes detectados en cada uno de los medios de compra (Onsite u Online)
  Para diagnosticar fraudes, se tienen 2 comandos:
  
  1. Validar fraudes onsite mediante el comando `make -C preparation/ onsite_state`
     
     El archivo de orquestación 'Makefile' identifica que se quieren validar de las compras realizadas, cuáles fueron fraudes onsite, para esto procede a invocar al archivo 'onsite_state.py' el cual a su vez busca en el archivo 'preparation.py' la función correspondiente a 'onsite_final_state'.
    En esta función se llama a las funciones de validación de compra onsite (onsite_transactions_validation) y esta a su vez a las funciones de carga (load_file_card) y limpieza de datos (cleansing_data) y posterior a ejecutarlas procede a 'diagnosticar fraudes'.
    Al diagnosticar, filtra las compras Onsite que tienen el atributo fraude = 1 y devuelve el mensaje "Se validan n transacciones fraudulentas Onsite".
    
    Imagen de ejecución:
    
    ![image](https://user-images.githubusercontent.com/17460738/221299235-af85e4b0-78f8-4666-b870-333bf757bb1a.png)
    
    De esta manera se diagnostican los fraudes Onsite.
    
   2. Validar fraudes online mediante el comando `make -C preparation/ online_state`
     
     El archivo de orquestación 'Makefile' identifica que se quieren validar de las compras realizadas, cuáles fueron fraudes online, para esto procede a invocar al archivo 'onsite_state.py' el cual a su vez busca en el archivo 'preparation.py' la función correspondiente a 'online_final_state'.
    En esta función se llama a las funciones de validación de compra online (online_transactions_validation) y esta a su vez a las funciones de carga (load_file_card) y limpieza de datos (cleansing_data) y posterior a ejecutarlas procede a 'diagnosticar fraudes'.
    Al diagnosticar, filtra las compras Online que tienen el atributo fraude = 1 y devuelve el mensaje "Se validan n transacciones fraudulentas Online".
    
    Imagen de ejecución:
    
    ![image](https://user-images.githubusercontent.com/17460738/221299602-59a5b8f2-12ae-4d2c-8741-7d05b40b8940.png)
    
    De esta manera se diagnostican los fraudes Onsite.


### **Repositorio en GitHub**

El código completo de este documento se encuentra disponible en:

https://github.com/mvalenciaar/productos_datos/tree/main

La documentación de todo el sistema de implementación se encuentra disponible en:

https://github.com/mvalenciaar/productos_datos/blob/main/README.md


### **Referencias**
[1] Legal Information Institute. “Fraude con tarjeta de crédito”. Disponible en: 
https://www.law.cornell.edu/wex/es/fraude_con_tarjeta_de_cr%C3%A9dito#:~:text=El
%20fraude%20con%20tarjeta%20de,forma%20de%20robo%20de%20identidad.

[2] BBC News. “Cómo se producen los fraudes con tarjetas de crédito y las reglas de 
oro para evitarlos”. 2017. Disponible en: https://www.bbc.com/mundo/vert-cap-40638275

[3] Asuntos Legales. “Responsabilidad bancaria en casos de fraudes electrónicos”. 
2019. Disponible en: https://www.asuntoslegales.com.co/consultorio/responsabilidad-bancaria-en-casos-de-fraudes-electronicos-288037

[4] Dinero. “¿Cuáles son las leyes que rigen a la banca digital y a la banca tradicional?”. 
2020. Disponible en: https://www.dinero.com/economia/articulo/legislacion-banca-cuales-son-las-leyes-que-rigen-a-la-banca-digital-y-a-la-banca-tradicional/28067


### **Entregable 2: Modelo analítico y Dashboard en Flask**

En esta etapa del trabajo, se construyen dos modelos analíticos para determinar la posibilidad de fraude de tarjetas de crédito, para ello, se decide entrenar un modelo de aprendizaje supervisado como es `Regresión Logística` y uno de clasificación como es `Árbol de Decisiones`, donde finalmente, se escogerá el mejor modelo de acuerdo a su `Accuracy`.

Posteriormente, se guardan las imágenes tanto del Análisis Exploratorio de los Datos (EDA) que se realiza, como de la matriz de confusión para cada modelo propuesto, y finalmente, la tabla con los accuracy de cada uno de ellos, obteniendo el mayor.

Esta información se lleva a una página `HTML`, la cual es construida por medio de la librería Flask de Python.
A continuación, se relacionan las imágenes de este dashboard.

**DASHBOARD - MODELOS ANALÍTICOS PARA FRAUDE EN TARJETAS DE CRÉDITO**

![EDA1](https://user-images.githubusercontent.com/56141354/226215768-53c5d9be-8572-4168-938c-8631e3642f8e.JPG)


![EDA2](https://user-images.githubusercontent.com/56141354/226215921-bdcd57a2-8761-456e-a278-8a28fc0be800.JPG)


![EDA3](https://user-images.githubusercontent.com/56141354/226215927-b9c66044-c97d-46e6-b07c-e25f69eb2423.JPG)


![Matriz1](https://user-images.githubusercontent.com/56141354/226215940-24101531-09be-4c55-9911-19bf667c8625.JPG)


![Matriz2](https://user-images.githubusercontent.com/56141354/226215944-ee0fdd29-2ea1-49f8-b25b-d0aeb61d24b7.JPG)


![tabl1](https://user-images.githubusercontent.com/56141354/226215951-df72056d-4c13-479e-92c0-90768f3df070.JPG)

### **Entregable 3: Modelo y ciclo de vida MLOps en MLFlow**

Para este entregable se realizan los pasos de ciclo de vida de un prodcuto de ML con la ayuda del paquete MLFlow.

Lo primero que se debe hacer es tomar uno de los modelos que teníamos en la entrega 2, para este caso tomamos el modelo de aprendizaje supervisado, `Regresión Logística` con el parámetro de `max_itrer` con un valor por defecto de 200. A este modelo se le realiza en proceso de empaquetado, con sus funciones respectivas de:

    
```python
#Carga de datos

def load_data():
    
    from preparation.preparation d

    ''' Defined Load File '''
    model_data = load_file_card().copy()

    ''' Preparin data for analytic model '''
    x = model_data.drop("fraud", axis = 1).values
    y = model_data["fraud"].values

    return x, y
```

```python
#Particionamiento de datos

def make_train_test_split(x, y):
    
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    
    smote = SMOTE(random_state=39)
    non_fraud_over, fraud_over = smote.fit_resample(x, y)

    non_fraud_over_df = pd.DataFrame(non_fraud_over, columns=["distance_from_home", "distance_from_last_transaction",
        "ratio_to_median_purchase_price", "repeat_retailer", "used_chip",
        "used_pin_number", "online_order"])

    non_fraud_over_df["fraud"] = fraud_over
    df3 = non_fraud_over_df

    feature_columns = ["distance_from_home", "distance_from_last_transaction",
    "ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number", "online_order"]

    X_smote = df3[feature_columns]
    y_smote = df3.fraud

    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=39)
    
    return X_train_smote, X_test_smote, y_train_smote, y_test_smote
```

```python
#Calculo de metricas

def eval_metrics(y_test_smote, y_pred_logreg_smote):
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    
    #confusion matrix
    confusion_matrix_logreg = confusion_matrix(y_test_smote, y_pred_logreg_smote)
    
    #classification report
    classification_report_logreg = classification_report(y_test_smote, y_pred_logreg_smote, digits=6)
    
    return confusion_matrix_logreg, classification_report_logreg  
```

```python
#Reporte de métricas

def report(confusion_matrix_logreg, classification_report_logreg):
    
    print(f"Confusion matrix Logistic Regression: {confusion_matrix_logreg}")
    print(f"classification report RL: {classification_report_logreg}")
```

```python
#Entrenamiento modelo

def train_logreg (max_iter = 200):
    
    import mlflow.sklearn
    import mlflow    
    from sklearn.linear_model import LogisticRegression
    
    #Habilita autolog
    mlflow.sklearn.autolog()
    
    
    x, y = load_data()
    
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = make_train_test_split(x, y)
    
    print('Tracking directory:', mlflow.get_tracking_uri())
    
    with mlflow.start_run(run_name = "LogReg_Model") as run:
    
        logreg = LogisticRegression(max_iter=max_iter)
        logreg.fit(X_train_smote, y_train_smote)

        y_pred_logreg_smote = logreg.predict(X_test_smote)
        
        
        confusion_matrix_logreg, classification_report_logreg = eval_metrics(y_test_smote, y_pred_logreg_smote)
        
        report(confusion_matrix_logreg, classification_report_logreg)
        
        #
        # Tracking de parámetros
        #
        mlflow.log_param("max_iter", max_iter)

        #
        # Tracking de metricas
        #
        mlflow.log_metric("accuracy_logreg", logreg.score(X_test_smote, y_test_smote))
        #mlflow.log_metric("classification_report_logreg", classification_report_logreg)

        #
        # Log del modelo
        #
        mlflow.sklearn.log_model(logreg, "model")
        
        return (run.info.experiment_id, run.info.run_id)
```

Luego de tener definido y estructurado el modelo, se hacen ejecuciones manuales buscando el mejor parámetro `max_iter`

```python
train_logreg()
```

```python
train_logreg(300)
```

```python
train_logreg(500)
```

```python
train_logreg(5000)
```

Una vez se tienen los resultados de estas ejecuciones, se escoge la mejor para hacer el registro del modelo, en este caso se tomó la por defecto, con el parámetro `max_iter = 200`

A partir de aquí se desarrollan las actividades solicitadas para el 3er entregable:

1. Registro del modelo en MLFlow:

Para hacer el registro del modelo en MLFlow, se necesita que las corridas estén respalados en una base de datos, para este caso usaremos SQLite. Para esto se hacen estas adiciones de funciones en el código:

```python
#Seteo de ruta para registro de modelos
def set_tracking_uri():

    import mlflow

    mlflow.set_tracking_uri('sqlite:///mlruns.db')
```

```python
#Visualización config de rutas
def display_config():

    import mlflow

    print("Current model registry uri: {}".format(mlflow.get_registry_uri()))
    print("      Current tracking uri: {}".format(mlflow.get_tracking_uri()))
```

Adicional, de modifica el código de la función del entrenamiento del modelo, para que llame la fn de cambio de tracking uri y para que registre la corrida como modelo, quedando así:

```python
#Entrenamiento modelo

def train_logreg (max_iter = 200):
    
    import mlflow.sklearn
    import mlflow    
    from sklearn.linear_model import LogisticRegression
    
    #Habilita autolog
    mlflow.sklearn.autolog()
    
    #Setea entorno para registros
    set_tracking_uri()
    
    x, y = load_data()
    
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = make_train_test_split(x, y)
    
    print('Tracking directory:', mlflow.get_tracking_uri())
    
    with mlflow.start_run(run_name = "LogReg_Model") as run:
    
        logreg = LogisticRegression(max_iter=max_iter)
        logreg.fit(X_train_smote, y_train_smote)

        y_pred_logreg_smote = logreg.predict(X_test_smote)
        
        
        confusion_matrix_logreg, classification_report_logreg = eval_metrics(y_test_smote, y_pred_logreg_smote)
        
        report(confusion_matrix_logreg, classification_report_logreg)
        
        #
        # Tracking de parámetros
        #
        mlflow.log_param("max_iter", max_iter)

        #
        # Tracking de metricas
        #
        mlflow.log_metric("accuracy_logreg", logreg.score(X_test_smote, y_test_smote))
        #mlflow.log_metric("classification_report_logreg", classification_report_logreg)

        #
        # Log del modelo
        #
        mlflow.sklearn.log_model(logreg, "model")
        
        #
        #Registro del modelo luego de varias corridas (se descomenta luego de correr con varias max_iter)
        #
        mlflow.register_model(
            f"runs:/{run.info.run_id}",
            f"sklearn-{max_iter}-iterations-logistic-regression-model"
        )
        
        return (run.info.experiment_id, run.info.run_id)
```

Una vez se hicieron estos cambios, se procede a ejecutar nuevamente la corrida con el parámetro `max_iter = 200`, la cual al finalizar habrá invocado la fn de `mlflow.register_model`

```python
train_logreg()
```

Esto dá como resultado el siguiente ouput:

![image](https://user-images.githubusercontent.com/17460738/227718423-b71a4ac7-0342-4935-9e89-652190bf4a7a.png)

Se observa que se registró exitosamente el modelo en MLFlow, con el nombre 'sklearn-200-iterations-logistic-regression' y el run_id = '541a266554654c8cbe981aa67a94256a'

Para verificarlo, se abre la consola de mlflow y se tiene:

![image](https://user-images.githubusercontent.com/17460738/227718465-c38457f8-15b5-4600-a0ab-06a8ffecf51d.png)

2. Paso del modelo a ambiente de producción (método por interfaz gráfica):

Para este paso, se tiene que se puede hacer desde la consola ui de mlflow, yendo al apartado de 'Models' y entrando al modelo que se quiere transicionar, en este caso el modelo `sklearn-200-iterations-logistic-regression`.

Luego de estar en él, se pasa entre ambientes, pasando primero por el ambiente 'Staging':

![image](https://user-images.githubusercontent.com/17460738/227718931-a1897c38-c611-4eea-902d-842ab14f7ec8.png)

Y quedando:

![image](https://user-images.githubusercontent.com/17460738/227718943-2c09fbf4-888e-4f57-9474-fdde92a36576.png)

Y finalmente, transicionando a producción:

![image](https://user-images.githubusercontent.com/17460738/227718967-80ff42c0-a215-49c9-823e-19d2f4fd305e.png)

Y quedando:

![image](https://user-images.githubusercontent.com/17460738/227718985-0a52eeb2-2c94-4259-800e-c57b2a333a67.png)

Con esto ya se tiene que el modelo se encuentra registrado y en el ambiente de producción en MLFlow.

3. Disponibilizar el modelo por medio de API y consumirlo:

Para culminar, se requiere 'servir' o disponibilizar el modelo por medio de una API REST, la cual el mismo MLFlow nos permite hacerlo.

Antes de disponibilizar el modelo, se debe crear una nueva función en python para la formación del set de pruebas como un parámetro llamado `data`, el cual estará formateado como json y se le pasará a la API para su consulta. La función es:

```python
def get_json_test_data():

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)

    data = x_test.iloc[0:10,:].to_json(orient='split')

    data = repr(data)
    return data

data = get_json_test_data()
```

Luego de tener el parámetro `data` pasamos a disponibilizar el modelo, para esto, abrimos una consola de terminal y ejecutamos el siguiente comando:

```
#!/usr/bin/env sh

export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

mlflow models serve --model-uri runs:/541a266554654c8cbe981aa67a94256a/model --env-manager=local
```

Esto arroja el output:

![image](https://user-images.githubusercontent.com/17460738/227719261-d8a0b6d7-19d0-4274-a3e2-f2819197fd71.png)

Con esto se evidencia que se encuentra ejecutando el server de MLFlow con el modelo disponible para consumo por API.

Para consumirlo, desde python, se ejecuta el siguiente comando, el cual consume el modelo y envía el parámetro `data`:

```python
!curl http://localhost:5000/invocations -H 'Content-Type: application/json' -d {data}
```

Al consumirlo, nos retorna el siguiente output:

![image](https://user-images.githubusercontent.com/17460738/227719693-cab833a6-5c28-46a1-b947-01a7c70d53f2.png)


Con esto vemos que se está consumiendo de manera exitosa el modelo como API y se dá por culminada la entrega.
