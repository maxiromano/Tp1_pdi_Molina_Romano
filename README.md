# Procesamiento de Imágenes I - Detección y Validación de Formularios

## Configuración del Entorno Virtual


1.  **Navega a la carpeta de tu proyecto:**
    ```bash
    cd /ruta/a/tu/proyecto
    ```

2.  **Crea el entorno virtual:**
    ```bash
    python -m venv venv
    ```
    Esto creará una carpeta llamada `venv` dentro de tu directorio de proyecto.

3.  **Activa el entorno virtual:**

    *   **En Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **En macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    Verás `(venv)` al inicio de tu línea de comandos, indicando que el entorno virtual está activo.

## Instalación de Dependencias

Con el entorno virtual activado, instala las librerías necesarias utilizando `pip`:

```bash
pip install numpy pandas matplotlib opencv-python