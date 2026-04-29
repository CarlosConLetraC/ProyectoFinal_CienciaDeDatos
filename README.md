# Airbnb Price Predictor: Documentación Técnica de Arquitectura

Este sistema es un motor de procesamiento concurrente de alto rendimiento diseñado para tareas de ciencia de datos intensivas. Implementa una arquitectura híbrida donde la eficiencia de **C++** se integra con la flexibilidad dinámica de **LuaJIT** y la capacidad analítica de **Python**.

## 1. Ecosistema de Herramientas de Shell e Interactividad

El proyecto abstrae la complejidad de las rutas y dependencias de LuaJIT mediante un conjunto de interfaces de línea de comandos (CLI) diseñadas para la depuración y la inspección de datos en tiempo real.

### 🖥 Interfaces de Ejecución

* **`./cmd` (Interface de Ejecución Directa)**:
    Permite ejecutar sentencias Lua aisladas directamente desde la terminal de Linux. Es ideal para inspeccionar el estado del sistema o probar funciones rápidas de las librerías cargadas.
    - *Ejemplo de uso*: `./cmd "system.print('Inspección de sistema', {{os}, 5})"`

* **`./initconsole` (Entorno REPL Interactivo)**:
    Lanza una consola interactiva (Read-Eval-Print Loop) con el núcleo del sistema (`import/init`) y librerías base (`Math`, `Table`, `system`) precargadas.
    - *Uso matemático*: `import('Vector3'); system.print(Vector3.one + Vector3.new(5,15,10))`
    - *Modo Scripting*: Soporta el flag `--exec` para ejecutar lógica compleja (como iteraciones de tablas) antes de entrar en modo interactivo.

* **`./runclient` (Lanzador de Aplicaciones)**:
    Punto de entrada para los scripts de producción (`program.main.lua`). Gestiona la carga automática de plugins y el sistema de módulos antes de iniciar el proceso principal.

---

## 2. Infraestructura de Automatización y DevOps

El proyecto garantiza portabilidad absoluta y consistencia de entorno mediante contenedores y scripts de orquestación.

### 🐳 Gestión de Contenedores (`pods.sh`)
Define funciones para estandarizar el flujo de trabajo con **Podman**:
* `podmanbuild`: Construye la imagen asegurando que todas las dependencias (GCC, LuaJIT, Python) estén presentes.
* `podmanrun`: Ejecuta el contenedor con `--userns=keep-id` y montajes de volumen `:Z` para permitir persistencia de datos con los permisos de usuario del host.

### 🏗 Setup y Compilación
* **`configurarentorno.sh`**: Script de bootstrap que configura el entorno virtual de Python (`venv`), instala dependencias de sistema y prepara el gestor de paquetes `luarocks`.
* **`build.sh`**: Punto de entrada para la compilación del binario `backend` en C++17, optimizado para el hardware local mediante la detección automática de núcleos.

---

## 3. Pipeline de Procesamiento de Datos (`run.sh`)

El archivo `run.sh` coordina el flujo completo de ciencia de datos:

1.  **Validación**: Verifica la presencia de LuaJIT y el entorno virtual de Python.
2.  **Particionamiento**: `program.main.lua` divide el dataset original en fragmentos (shards) y genera una cola de trabajos en `daemons/`.
3.  **Procesamiento Concurrente (`backend`)**: El binario de C++ distribuye los shards entre hilos paralelos. Cada hilo ejecuta un worker LuaJIT para limpieza (`NaN`), normalización y **One-Hot Encoding**.
4.  **Consolidación**: `merge.lua` reúne los resultados procesados en un único dataset listo para el modelo.
5.  **Entrenamiento**: El motor C++ calcula la Regresión Lineal Múltiple y exporta los pesos a `model_final.json`.
6.  **Visualización**: Se activan scripts de Python para generar gráficas de diagnóstico en el directorio `/plots`.

---

## 4. Resultados y Métricas del Modelo

Basado en el procesamiento de **74,113 registros**, el sistema reporta:

| Métrica | Valor |
| :--- | :--- |
| **$R^2$ (Coeficiente de Determinación)** | **0.5281** |
| **MSE (Error Cuadrático Medio)** | **0.2439** |
| **MAE (Error Absoluto Medio)** | **0.3720** |

### Hallazgos de Análisis
* **Impacto de Privacidad**: La variable `room_shared` es el predictor con el impacto negativo más fuerte (**-0.742**).
* **Capacidad Física**: El número de habitaciones (`bedrooms`: +0.146) y baños (`bathrooms`: +0.133) muestran una correlación positiva directa con el precio.

---

## 🛠 Instrucciones de Uso (Flujo Completo)

1. **Cargar funciones**: `source pods.sh`
2. **Construir entorno**: `podmanbuild Dockerfile airbnb-proj`
3. **Ejecutar contenedor**: `podmanrun airbnb-proj`
4. **Ejecutar experimento**: `./run.sh`

---

## Instalación del software
```bash
git clone --recursive https://github.com/CarlosConLetraC/ProyectoFinal_CienciaDeDatos.git
cd ProyectoFinal_CienciaDeDatos
chmod +x initconsole cmd runclient *.sh
```

---

# Configurar el entorno (recomendado usar podman)
```bash
./configurarentorno.sh
```
---

# Compilar backend.cpp
```bash
./build.sh
```
---

# Ejecutar proyecto completo
```bash
./run.sh
```

---