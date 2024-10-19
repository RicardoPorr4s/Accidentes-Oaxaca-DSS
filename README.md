# Dashboard de Accidentes Viales en Oaxaca

## Descripción
Este proyecto consiste en un dashboard interactivo que permite visualizar y analizar datos sobre accidentes viales en Oaxaca. A través de un enfoque basado en datos, se busca evaluar patrones y tendencias en la seguridad vial de la región.

## Características
- **Interactividad**: Los usuarios pueden explorar datos a través de gráficos y visualizaciones interactivas, seleccionando el municipio que desean consultar.
- **Análisis de Datos**: Aplicación de KPIs y modelos matemáticos para entender los factores que influyen en los accidentes viales por año, mes, día, tipo de accidente, etc.
- **Proceso ETL**: Implementación de un proceso de Extracción, Transformación y Carga (ETL) para preparar y limpiar los datos.

## Fuentes de Datos
Los datos utilizados en este proyecto fueron obtenidos del **Instituto Nacional de Estadística y Geografía (INEGI)**, una fuente confiable de información estadística en México.
Gobierno de México. (2021). Accidentes de tránsito terrestre en zonas urbanas y suburbanas. Datos.gob.mx. https://datos.gob.mx/busca/dataset/accidentes-de-transito-terrestre-en-zonas-urbanas-y-suburbanas1

## Tecnologías Utilizadas
- **Lenguaje**: Python
- **Base de datos**: MySQL 8.0.36
- **Bibliotecas**: 
  - Dash
  - Plotly
  - Pandas
  - SQLAlchemy
  - Statsmodels
  - Scikit-learn

## Instalación
Para ejecutar este proyecto, asegúrate de tener Python y MySQL instalados. Luego, puedes instalar las dependencias necesarias con:

```bash
pip install -r requirements.txt
