# Detección de Aves con YOLOv8 
Proyecto de la materia de Visión Artificial 
Entrenamiento de un modelo de la familia YOLO para la detección de aves en imágenes, junto con una propuesta de integración del modelo en un entorno real/industrial.

## Integrantes
- **Marco Antonio Gil Martinez** — CETI (Centro de Enseñanza Técnica Industrial), Mecatrónica, 6to Semestre — Registro 23310393
- **Elías Uriel Torres Hernández** — CETI (Centro de Enseñanza Técnica Industrial), Mecatrónica, 6to Semestre — Registro 23310395
---

## Tabla de contenido
1. [Descripción del proyecto](#descripción-del-proyecto)
2. [Dataset utilizado](#dataset-utilizado)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Instrucciones para correr el proyecto](#instrucciones-para-correr-el-proyecto)
5. [Resultados del modelo](#resultados-del-modelo)
6. [Caso de Estudio: Aplicación en la vida real](#caso-de-estudio-aplicación-en-la-vida-real)
---

## Descripción del proyecto
Se entrenó un modelo **YOLOv8 (nano)** para detectar aves en imágenes, usando un dataset público anotado en formato YOLO. El objetivo académico es:
1. Aplicar el flujo completo de un proyecto de Visión Artificial: preparación de dataset → entrenamiento → validación → inferencia.
2. Documentar cómo este mismo modelo podría integrarse a un sistema real de detección de aves en un entorno industrial (ver Caso de Estudio).

## Dataset utilizado
- **Nombre:** Birds Dataset (Roboflow Universe, autor: *kurisnis*)
- **Fuente:** https://universe.roboflow.com/kurisnis/birds-olyak
- **Contenido:** 277 imágenes anotadas, clase única `bird`, formato YOLOv8 (`images/` + `labels/` + `data.yaml`)
- **Licencia:** CC BY 4.0
Se eligió este dataset por venir ya etiquetado correctamente en formato YOLO, evitando errores de preprocesamiento manual y permitiendo enfocar el esfuerzo en el entrenamiento y la aplicación práctica del modelo.

## Estructura del repositorio
```
bird-yolo-project/
├── notebooks/
│   └── entrenamiento_yolo_aves.ipynb   # Notebook completo: descarga, entrenamiento, evaluación e inferencia
├── scripts/
│   ├── preprocesamiento.py             # Valida la integridad del dataset antes de entrenar
│   └── inferencia.py                   # Corre el modelo entrenado sobre imágenes nuevas
├── evidencias/
│   ├── predicciones/                   # Imágenes de prueba con bounding boxes generados
│   └── results.png                     # Curvas de entrenamiento (loss, mAP, precisión, recall)
├── requirements.txt
├── .gitignore
└── README.md
```

## Instrucciones para correr el proyecto
### Opción A — Entrenar desde cero (Google Colab, recomendado)
1. Abre `notebooks/entrenamiento_yolo_aves.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Activa GPU: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU`.
3. Crea una cuenta gratuita en [Roboflow](https://roboflow.com) y obtén tu API key.
4. Corre las celdas en orden. El notebook descarga el dataset, entrena el modelo, lo evalúa y genera las imágenes de evidencia con bounding boxes.
5. Descarga los resultados generados en `runs_aves/predicciones_evidencias/` y `runs_aves/yolov8n_aves/results.png`, y colócalos en la carpeta `evidencias/` de este repositorio.

### Opción B — Correr en local con un modelo ya entrenado
```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd bird-yolo-project
# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
# 3. (Opcional) Validar la integridad de un dataset descargado
python scripts/preprocesamiento.py --dataset ruta/al/dataset
# 4. Correr inferencia sobre imágenes nuevas usando el modelo entrenado
python scripts/inferencia.py --modelo modelo_aves_best.pt --imagenes ruta/a/imagenes --salida evidencias
```

## Resultados del modelo
| Métrica | Valor |
|---|---|
| mAP50 | *(completar tras correr el notebook)* |
| mAP50-95 | *(completar tras correr el notebook)* |
| Precisión | *(completar tras correr el notebook)* |
| Recall | *(completar tras correr el notebook)* |

> Las métricas finales y las curvas de entrenamiento (`results.png`) se generan automáticamente al correr el notebook y deben colocarse en `evidencias/`.
---

## Caso de Estudio: Aplicación en la vida real
### Título: Sistema de Detección Aviar para Prevención de Colisiones en Aeropuertos (Bird Strike Prevention System)
### 1. Problema a resolver
Las colisiones entre aves y aeronaves (*bird strikes*) son un riesgo real y costoso en la industria aeronáutica: pueden dañar motores, fuselaje y, en casos extremos, poner en riesgo la seguridad del vuelo. Los aeropuertos —incluyendo el Aeropuerto Internacional de Guadalajara (GDL), cercano a zonas con presencia de aves migratorias y fauna local— necesitan monitorear constantemente las pistas y zonas de aproximación para detectar aves antes de que represente un riesgo para el despegue o aterrizaje.

Actualmente, gran parte de este monitoreo depende de personal humano observando visualmente las pistas, lo cual es:
- **Lento** para reaccionar ante bandadas que aparecen repentinamente.
- **Inconsistente**, sobre todo en condiciones de baja visibilidad, lluvia o de noche.
- **Costoso** si se requiere personal dedicado las 24 horas en múltiples puntos de la pista.

Un sistema basado en YOLO permitiría automatizar la detección de aves en tiempo real y disparar alertas o acciones preventivas mucho más rápido que la observación humana.

### 2. Descripción del hardware propuesto
| Componente | Función | Propuesta |
|---|---|---|
| **Cámaras** | Captura de video continuo de las zonas críticas de pista y aproximación | Cámaras IP fijas tipo *bullet* con visión nocturna (infrarrojo) y resolución mínima 1080p, distribuidas cada ~300-500 m a lo largo de las pistas y zonas de aproximación. Complementadas con cámaras térmicas en puntos clave para detección nocturna o con poca visibilidad. |
| **Procesador de borde (edge)** | Correr el modelo YOLO localmente, sin depender de la nube, para minimizar la latencia | Un mini-PC o módulo embebido tipo **NVIDIA Jetson Orin Nano** (o similar) conectado a cada grupo de 2-3 cámaras. Tiene GPU integrada suficiente para correr YOLOv8n en tiempo real (>20 FPS). |
| **Servidor central** | Recibir las alertas de todos los nodos edge, centralizar el monitoreo y mostrar el dashboard a los operadores de torre de control | Servidor on-premise en el centro de operaciones del aeropuerto, conectado por red Ethernet/fibra a los nodos Jetson distribuidos en pista. |
| **Sistema de disuasión** | Actuar cuando se detecta una bandada de riesgo | Cañones de gas (*propane cannons*) automatizados, parlantes con sonidos de depredadores/distress calls, y drones disuasorios — todos activables remotamente desde el sistema central. |
| **Interfaz de torre de control** | Mostrar alertas visuales a los controladores aéreos | Pantalla/dashboard con mapa de la pista, ubicación de detecciones, nivel de riesgo (cantidad de aves, cercanía a la pista activa) y video en vivo con bounding boxes. |

### 3. Flujo de funcionamiento
1. **Captura continua:** las cámaras IP/térmicas distribuidas en pista capturan video 24/7 de las zonas de aproximación y despegue.
2. **Inferencia en el borde:** cada nodo Jetson corre el modelo YOLO entrenado sobre el stream de su cámara en tiempo real, detectando y contando aves en el cuadro (bounding boxes + nivel de confianza).
3. **Clasificación de riesgo:** un módulo de lógica simple evalúa la detección según:
   - Número de aves detectadas (bandada vs. individuo aislado).
   - Cercanía a la pista activa (zona de aproximación/despegue vs. zona perimetral).
   - Persistencia (¿siguen ahí después de varios frames, o fue una detección momentánea?).
4. **Generación de alerta:** si el nivel de riesgo supera el umbral configurado, el nodo edge envía una alerta al servidor central (vía red local) con: ubicación, cantidad estimada de aves, captura de imagen con bounding boxes y timestamp.
5. **Visualización en torre de control:** el dashboard central muestra la alerta en tiempo real sobre un mapa de la pista, junto con el video correspondiente, para que el controlador aéreo decida si retrasa una operación de despegue/aterrizaje.
6. **Acción automática de disuasión (opcional):** si el riesgo es alto y está configurado en modo automático, el sistema activa el dispositivo disuasorio más cercano a la zona detectada (cañón de gas o sonido de distress) para dispersar la bandada antes de intervención humana.
7. **Registro histórico:** cada detección se guarda en una base de datos con fecha, hora, ubicación y nivel de riesgo, generando un histórico útil para identificar patrones (por ejemplo, horarios o temporadas migratorias de mayor actividad aviar), lo cual permite mejorar la prevención a futuro y, potencialmente, reentrenar el modelo con nuevos datos reales capturados en el propio aeropuerto.

### 4. Justificación de YOLO para este caso
YOLO es ideal para esta aplicación porque:
- Procesa imágenes completas en una sola pasada (*you only look once*), lo que permite **detección en tiempo real** a alta velocidad de cuadro, crítico cuando se trata de segundos de diferencia para emitir una alerta.
- Las versiones ligeras (como `yolov8n`) corren eficientemente en hardware embebido (Jetson), evitando depender de conexión constante a la nube.
- Es robusto detectando múltiples objetos pequeños en el cuadro (bandadas completas), no solo un ave aislada.

### 5. Limitaciones y trabajo futuro
- El modelo actual fue entrenado con un dataset genérico de aves; para producción real se necesitaría reentrenar con imágenes capturadas en el propio aeropuerto (distintas especies locales, ángulos de cámara reales, condiciones de luz específicas).
- Sería necesario añadir clasificación por especie (no solo "ave" genérica), ya que el riesgo varía según el tamaño y comportamiento de cada especie.
- Se recomendaría complementar el sistema con radar aviar para detección a mayor distancia, usando YOLO como confirmación visual una vez que el radar detecta una bandada acercándose.
---

## Licencia y créditos
- Dataset: [Birds Dataset by kurisnis](https://universe.roboflow.com/kurisnis/birds-olyak) — CC BY 4.0
- Framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)