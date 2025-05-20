# Classic vs. AI 3D Reconstructor

Reconstrucción 3D a partir de imágenes 2D, comparando enfoques clásicos con métodos basados en Inteligencia Artificial.

## Descripción General

Este proyecto explora y compara dos enfoques principales para la reconstrucción de escenas u objetos 3D a partir de un conjunto de imágenes 2D:

1.  **Reconstrucción Clásica:** Utiliza técnicas tradicionales de fotogrametría, como Structure from Motion (SfM) y Multi-View Stereo (MVS).
2.  **Reconstrucción Basada en IA:** Emplea modelos de aprendizaje profundo para tareas como la estimación de profundidad y la segmentación de objetos, que luego se utilizan para generar la reconstrucción 3D.

El objetivo es analizar las fortalezas, debilidades y resultados de cada método.

## Estructura del Repositorio

A continuación, se describe la organización de los archivos y directorios principales que se encuentran en este repositorio de GitHub:

```
.
├── LICENSE                 # Archivo de licencia del proyecto
├── model.obj               # Ejemplo de un modelo 3D reconstruido (formato .obj)
├── README.md               # Este archivo
├── images/                 # Directorio con conjuntos de imágenes de entrada de ejemplo
│   ├── 0/
│   └── 1/
├── models/                 # Modelos de IA preentrenados
│   ├── dpt_hybrid.pt       # Modelo DPT (Dense Prediction Transformer) para estimación de profundidad
│   └── sam_vit_t.pth       # Modelo SAM (Segment Anything Model)
└── src/                    # Código fuente del proyecto
    ├── ai_depth.py         # Scripts para la estimación de profundidad usando modelos de IA
    ├── compare.py          # Herramientas o scripts para comparar los resultados de ambos métodos
    ├── fuse_ai.py          # Scripts para fusionar información de IA (ej. nubes de puntos a partir de mapas de profundidad)
    ├── reconstruct_classic_automatic.py # Script para la reconstrucción clásica de forma automatizada (posiblemente usando COLMAP)
    ├── reconstruct_classic.py  # Script para la reconstrucción clásica (posiblemente con pasos manuales o configurables)
    ├── viewer.py           # Visor para modelos 3D o nubes de puntos
    └── others/             # Otros scripts o utilidades experimentales
        ├── projecte.py
        ├── projecte2.py
        └── projecte3.py
```

**Nota:** El directorio `workspace/` (que puede contener bases de datos, resultados intermedios densos/dispersos de COLMAP, etc.) generalmente no se incluye en el repositorio y debe ser añadido a `.gitignore`.

## Módulos Principales

### 1. Reconstrucción Clásica
   - Implementada en `src/reconstruct_classic.py` y `src/reconstruct_classic_automatic.py`.
   - Probablemente utiliza algoritmos de Structure from Motion (SfM) para estimar la pose de la cámara y una nube de puntos dispersa, seguido de Multi-View Stereo (MVS) para generar una nube de puntos densa y una malla.
   - La versión automática podría interactuar con herramientas como COLMAP.

### 2. Reconstrucción Basada en IA
   - **Estimación de Profundidad:** `src/ai_depth.py` utiliza modelos como DPT (cargado desde `models/dpt_hybrid.pt`) para estimar mapas de profundidad para cada imagen de entrada.
   - **Segmentación (Opcional):** El modelo SAM (`models/sam_vit_t.pth`) podría usarse para segmentar objetos de interés antes de la reconstrucción.
   - **Fusión y Creación de Malla:** `src/fuse_ai.py` se encarga de tomar los mapas de profundidad (y posiblemente máscaras de segmentación) para generar una nube de puntos 3D o una malla.

### 3. Comparación y Visualización
   - `src/compare.py`: Podría contener métricas o métodos para evaluar y comparar la calidad de las reconstrucciones de ambos enfoques.
   - `src/viewer.py`: Una utilidad para visualizar los modelos 3D generados (ej. archivos `.obj` o nubes de puntos).

## Configuración del Entorno

Se recomienda encarecidamente utilizar un gestor de entornos como Conda para manejar las dependencias del proyecto, especialmente debido a la complejidad de algunas librerías y la necesidad de COLMAP.

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu_usuario/classic-vs-AI-3D-reconstructor.git
    cd classic-vs-AI-3D-reconstructor
    ```

2.  **Instalar Conda:**
    Si aún no tienes Conda, descárgalo e instálalo desde [Anaconda Distribution](https://www.anaconda.com/products/distribution) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

3.  **Crear y Activar Entorno Conda:**
    (Idealmente, se proporcionaría un archivo `environment.yml` para una configuración más sencilla)
    ```bash
    conda create -n reconstruct3d python=3.12  # O la versión de Python que prefieras/necesites
    conda activate reconstruct3d
    ```

4.  **Instalar COLMAP:**
    COLMAP es una herramienta fundamental para la parte de reconstrucción clásica (Structure from Motion y Multi-View Stereo).
    *   **Windows:** Puedes descargar los ejecutables precompilados desde la [página de releases de COLMAP](https://github.com/colmap/colmap/releases). Asegúrate de añadir el directorio de COLMAP (que contiene `colmap.bat` o `colmap.exe`) a tu PATH del sistema o del entorno Conda.
    *   **Linux/macOS:** Puedes compilarlo desde las fuentes o instalarlo mediante gestores de paquetes si está disponible. La compilación desde fuentes requiere un compilador de C++ (como GCC o Clang) y varias librerías de desarrollo (Ceres Solver, Qt, etc.). Sigue las [instrucciones oficiales de instalación de COLMAP](https://colmap.github.io/install.html).
        ```bash
        # Ejemplo para Ubuntu (puede variar según la distribución y versión)
        sudo apt-get install \
            git \
            cmake \
            ninja-build \
            build-essential \
            libboost-program-options-dev \
            libboost-filesystem-dev \
            libboost-graph-dev \
            libboost-regex-dev \
            libboost-system-dev \
            libboost-test-dev \
            libeigen3-dev \
            libsuitesparse-dev \
            libfreeimage-dev \
            libmetis-dev \
            libgoogle-glog-dev \
            libgflags-dev \
            libglew-dev \
            qtbase5-dev \
            libqt5opengl5-dev \
            libcgal-dev \
            libcgal-qt5-dev
        # Clonar y compilar COLMAP (consultar la guía oficial para los pasos más actualizados)
        # git clone https://github.com/colmap/colmap.git
        # cd colmap
        # mkdir build && cd build
        # cmake .. -GNinja
        # ninja
        # sudo ninja install
        ```
    *   **Verificación:** Después de la instalación, deberías poder ejecutar `colmap -h` en tu terminal.

5.  **Dependencias de Python:**
    Una vez que COLMAP esté configurado y el entorno Conda activado, instala las librerías de Python.
    - OpenCV (`cv2`)
    - NumPy
    - PyTorch (para los modelos de IA) - Asegúrate de instalar la versión compatible con tu hardware (CPU o GPU CUDA).
    - Open3D (para manipulación y visualización de nubes de puntos/mallas)
    - Matplotlib

    Puedes instalarlas usando `pip` o `conda install` (preferiblemente `conda install` para bibliotecas complejas como PyTorch para asegurar la compatibilidad con CUDA si es necesario).
    ```bash
    # Ejemplo con conda para PyTorch (visita pytorch.org para el comando exacto según tu sistema y CUDA)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # Ejemplo para CUDA 11.8

    # Usando pip para el resto (o busca equivalentes en conda)
    pip install opencv-python numpy open3d matplotlib
    # ... y otras dependencias que puedan ser necesarias (idealmente listadas en un requirements.txt).
    ```
    Si existe un archivo `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

6.  **Compilador de C++ (para dependencias de librerías Python o COLMAP si se compila desde fuente):**
    Algunas librerías de Python pueden tener extensiones escritas en C/C++ y necesitarán un compilador para instalarse desde el código fuente (si no hay un wheel precompilado disponible para tu sistema/versión de Python).
    *   **Windows:** Instala las "Herramientas de compilación de C++ de Microsoft Visual Studio" (Build Tools for Visual Studio).
    *   **Linux:** Instala `build-essential` (ej. `sudo apt-get install build-essential`).
    *   **macOS:** Instala las herramientas de línea de comandos de Xcode (`xcode-select --install`).
    Esto también es crucial si decides compilar COLMAP desde las fuentes.

7.  **Modelos de IA:**
    Los modelos (`dpt_hybrid.pt`, `sam_vit_t.pth`) se proporcionan en el directorio `models/`. Si fueran demasiado grandes para el repositorio, aquí se indicarían los enlaces de descarga y dónde colocarlos.

## Uso

Las instrucciones detalladas sobre cómo ejecutar cada script se añadirían aquí. Por ejemplo:

### Ejecutar Reconstrucción Clásica (Automática)
```bash
python src/reconstruct_classic_automatic.py --images_path images/0/ --output_path workspace/classic_output/
```

### Ejecutar Estimación de Profundidad con IA
```bash
python src/ai_depth.py --images_path images/0/ --model_path models/dpt_hybrid.pt --output_depth_maps workspace/ai_output/depth_maps/
```

### Fusionar Resultados de IA para crear Nube de Puntos
```bash
python src/fuse_ai.py --depth_maps_path workspace/ai_output/depth_maps/ --output_ply workspace/ai_output/point_cloud.ply
```

### Visualizar un Modelo
```bash
python src/viewer.py --model_path model.obj
# o
python src/viewer.py --model_path workspace/ai_output/point_cloud.ply
```
*(Estos son ejemplos, los argumentos y scripts reales pueden variar)*

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un *issue* para discutir cambios importantes o envía un *pull request*.

## Licencia

Este proyecto está bajo la Licencia especificada en el archivo `LICENSE`.
