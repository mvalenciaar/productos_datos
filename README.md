# **Trabajo final Productos de Datos**

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

