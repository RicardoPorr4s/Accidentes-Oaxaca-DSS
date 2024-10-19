import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objs as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import glm
# Conectar a la base de datos
username = 'root'
password = 'Rycardo-Porr4s'
server = 'localhost'
database = 'accidentes'
engine = create_engine(f'mysql+pymysql://{username}:{password}@{server}/{database}')

# Leer los datos
query = '''
SELECT a.id_accidente, t.anio, t.mes, t.diasemana, t.dia_fes, t.fin_semana_largo,
       u.nom_municipio, v.tipaccid, v.automovil, v.campasaj, v.microbus, 
       v.pascamion, v.omnibus, v.tranvia, v.camioneta, v.camion, v.tractor, 
       v.ferrocarril, v.motociclet, v.bicicleta, v.otrovehic,
       c.causaacci, c.caparod,
       a.clasacc
FROM accidente a
JOIN tiempo t ON a.id_tiempo = t.id_tiempo
JOIN ubicacion u ON a.id_ubicacion = u.id_ubicacion
JOIN vehiculo v ON a.id_vehiculo = v.id_vehiculo
JOIN causa_accidente c ON a.id_causa = c.id_causa
'''
df = pd.read_sql(query, engine)

# Preprocesamiento de datos
df['fecha'] = pd.to_datetime(df['anio'].astype(str) + '-' + df['mes'].astype(str) + '-01')
df = df.sort_values('fecha')

# Filtrar los registros para excluir 'Certificado cero'
df = df[df['diasemana'] != 'Certificado cero']

# Crear una columna con el número total de accidentes por día
vehiculo_cols = ['automovil', 'campasaj', 'microbus', 'pascamion', 'omnibus', 'tranvia', 
                 'camioneta', 'camion', 'tractor', 'ferrocarril', 'motociclet', 
                 'bicicleta', 'otrovehic']
df['Total_Accidentes'] = df[vehiculo_cols].sum(axis=1)

# Clasificar los días como laborales, festivos o fines de semana largos
def classify_day(row):
    if row['fin_semana_largo'] == 1:
        return 'Fin de Semana Largo'
    elif row['dia_fes'] != 0:
        return 'Festivo'
    else:
        return 'Laboral'

df['Tipo_Dia'] = df.apply(classify_day, axis=1)

# Ordenar los días de la semana
days_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df['diasemana'] = pd.Categorical(df['diasemana'], categories=days_order, ordered=True)

# Crear una lista de opciones para el dropdown de municipios
municipios_options = [{'label': municipio, 'value': municipio} for municipio in df['nom_municipio'].unique()]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="Dashboard de Accidentes de Tráfico en Oaxaca",
        color="dark",
        dark=True,
    ),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Seleccionar Municipio"),
                dcc.Dropdown(
                    id='municipio-selector',
                    options=municipios_options,
                    value=municipios_options[0]['value'],
                    style={'color': 'black'}
                )
            ], width=12)
        ], className='chart-row', style={'margin-bottom': '20px'}),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Número Total de Accidentes por Año"),
                    dcc.Graph(id='accidents-per-year', style={'border': '2px solid white', 'padding': '10px'}) 
                ], className='chart-container'),
                html.P("Este gráfico muestra el número total de accidentes ocurridos cada año. Utiliza modelos ARIMA y Holt-Winters para predecir los accidentes en los próximos años.")
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H2("Número de Accidentes por Mes"),
                    dcc.Dropdown(
                        id='period-selector',
                        options=[          
                            {'label': 'Últimos 24 meses', 'value': 24},
                            {'label': 'Últimos 36 meses', 'value': 36},
                            {'label': 'Últimos 48 meses', 'value': 48},
                        ],
                        value=24, style={'color': 'black'}
                    ),
                    dcc.Graph(id='accidents-per-month', style={'border': '2px solid white', 'padding': '10px'}) 
                ], className='chart-container'),
                html.P("Este gráfico muestra la distribución mensual de los accidentes. Utiliza el modelo Holt-Winters para realizar previsiones.")
            ], width=6),
        ], className='chart-row'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Número de Accidentes por Día de la Semana"),
                    dcc.Graph(id='accidents-per-day', style={'border': '2px solid white', 'padding': '10px'})  
                ], className='chart-container'),
                html.P("Este gráfico muestra la media de accidentes que ocurren en cada día de la semana.")
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H2("Número de Accidentes por Tipo de Día"),
                    dcc.Graph(id='accidents-per-type-day', style={'border': '2px solid white', 'padding': '10px'})  
                ], className='chart-container'),
                html.P("Este gráfico muestra la media de accidentes según el tipo de día: laboral, festivo o fin de semana largo.")
            ], width=6),
        ], className='chart-row'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Causa del Accidente"),
                    dcc.Graph(id='accidents-by-cause', style={'border': '2px solid white', 'padding': '10px'})  
                ], className='chart-container'),
                html.P("Este gráfico muestra la distribución de los accidentes según su causa.")
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H2("Accidentes por Tipo de Vehículo"),
                    dcc.Graph(id='accidents-vehicle', style={'border': '2px solid white', 'padding': '10px'})  
                ], className='chart-container'),
                html.P("Este gráfico muestra la distribución de accidentes según el tipo de vehículo involucrado.")
            ], width=6),
        ], className='chart-row'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Número de Accidentes por Tipo de Accidente"),
                    dcc.Graph(id='accidents-by-type', style={'border': '2px solid white', 'padding': '10px'})  
                ], className='chart-container'),
                html.P("Este gráfico muestra la distribución de accidentes según el tipo de accidente ocurrido.")
            ], width=6)
        ], className='chart-row'),
    ], className='main-container')
], style={'backgroundColor': 'black', 'color': 'white', 'padding': '20px'})  

# Callbacks para los gráficos

@app.callback(
    Output('accidents-per-year', 'figure'),
    Input('municipio-selector', 'value')
)
def update_accidents_per_year(selected_municipio):
    filtered_df = df[df['nom_municipio'] == selected_municipio]
    annual_data = filtered_df.groupby('anio').size().reset_index(name='accidents')   
    hw_model = ExponentialSmoothing(annual_data['accidents'], seasonal='add', seasonal_periods=3).fit()
    hw_forecast = hw_model.forecast(3)
    forecast_years = pd.DataFrame({
        'anio': range(annual_data['anio'].max() + 1, annual_data['anio'].max() + 4),
        'HW_Forecast': hw_forecast
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=annual_data['anio'], y=annual_data['accidents'],
                             mode='lines+markers', name='Accidentes'))
    fig.add_trace(go.Scatter(x=forecast_years['anio'], y=forecast_years['HW_Forecast'],
                             mode='lines+markers', name='Predicción'))
    fig.update_layout(title='Número de Accidentes por Año', xaxis_title='Año', yaxis_title='Número de Accidentes')
    return fig

@app.callback(
    Output('accidents-per-month', 'figure'),
    [Input('period-selector', 'value'),
     Input('municipio-selector', 'value')]
)
def update_accidents_per_month(period, selected_municipio):
    filtered_df = df[df['nom_municipio'] == selected_municipio]
    monthly_data = filtered_df.groupby('fecha').size().reset_index(name='accidents')
    last_n_months = monthly_data.tail(period)
    hw_model = ExponentialSmoothing(last_n_months['accidents'], seasonal='add', seasonal_periods=12).fit()
    hw_forecast = hw_model.forecast(12)
    forecast_months = pd.DataFrame({
        'fecha': pd.date_range(start=last_n_months['fecha'].max(), periods=12, freq='MS'),
        'HW_Forecast': hw_forecast
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_n_months['fecha'], y=last_n_months['accidents'],
                             mode='lines+markers', name='Accidentes'))
    fig.add_trace(go.Scatter(x=forecast_months['fecha'], y=forecast_months['HW_Forecast'],
                             mode='lines+markers', name='Predicción'))
    fig.update_layout(title='Número de Accidentes por Mes', xaxis_title='Mes', yaxis_title='Número de Accidentes')
    return fig

@app.callback(
    Output('accidents-per-day', 'figure'),
    Input('municipio-selector', 'value')
)
def update_accidents_per_day(selected_municipio):
    if selected_municipio is None:
        return go.Figure()

    filtered_df = df[df['nom_municipio'] == selected_municipio]
    
    # Asegurarse de que vehiculo_cols existe y no está vacío
    if 'vehiculo_cols' not in globals() or not vehiculo_cols:
        raise ValueError("La variable 'vehiculo_cols' no está definida o está vacía.")
    
    # Calcular el total de accidentes
    filtered_df['Total_Accidentes'] = filtered_df[vehiculo_cols].sum(axis=1)
    
    # Definir el orden de los días de la semana
    days_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    filtered_df['diasemana'] = pd.Categorical(filtered_df['diasemana'], categories=days_order, ordered=True)
    
    # Ajustar el modelo de regresión y realizar ANOVA
    model = ols('Total_Accidentes ~ C(diasemana)', data=filtered_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    # Calcular la media de accidentes por día de la semana
    mean_accidents = filtered_df.groupby('diasemana')['Total_Accidentes'].mean().reset_index()
    
    # Calcular el total de accidentes por día de la semana
    total_accidents = filtered_df.groupby('diasemana')['Total_Accidentes'].sum().reset_index()

    # Crear la figura combinada
    fig = go.Figure()

    # Agregar las barras del total de accidentes
    fig.add_trace(
        go.Bar(
            x=total_accidents['diasemana'],
            y=total_accidents['Total_Accidentes'],
            name='Total de Accidentes',
            marker=dict(color='rgba(58, 99, 224, 0.6)', line=dict(color='black', width=1))  # Color azul
                    )
    )

    # Agregar la línea de la media de accidentes
    fig.add_trace(
        go.Scatter(
            x=mean_accidents['diasemana'],
            y=mean_accidents['Total_Accidentes'],
            name='Media de Accidentes',
            mode='lines+markers',
            line=dict(color='red'),
            marker=dict(size=10)
        )
    )

    # Actualizar el layout de la figura
    fig.update_layout(
        title='Total y Media de Accidentes por Día de la Semana',
        xaxis_title='Día de la Semana',
        yaxis_title='Número de Accidentes',
        barmode='group'
    )

    return fig


@app.callback(
    Output('accidents-per-type-day', 'figure'),
    Input('municipio-selector', 'value')
)
def update_accidents_per_type_day(selected_municipio):
    filtered_df = df[df['nom_municipio'] == selected_municipio]
    filtered_df['Total_Accidentes'] = filtered_df[vehiculo_cols].sum(axis=1)
    
    def classify_day(row):
        if row['fin_semana_largo'] == '1': 
            return 'Fin de Semana Largo'
        elif row['dia_fes'] != '0':  
            return 'Festivo'
        else:
            return 'Laboral'
    
    filtered_df['Tipo_Dia'] = filtered_df.apply(classify_day, axis=1)
    
    # Verificar que hay suficientes datos para cada tipo de día
    tipos_dia = filtered_df['Tipo_Dia'].value_counts()
    if any(tipos_dia < 2):
        # Si no hay suficientes datos, solo mostrar la gráfica de puntos
        mean_accidents = filtered_df.groupby('Tipo_Dia')['Total_Accidentes'].mean().reset_index()
        tipo_dia_order = ['Laboral', 'Festivo', 'Fin de Semana Largo']
        mean_accidents['Tipo_Dia'] = pd.Categorical(mean_accidents['Tipo_Dia'], categories=tipo_dia_order, ordered=True)
        mean_accidents = mean_accidents.sort_values('Tipo_Dia')

        fig = px.scatter(mean_accidents, x='Tipo_Dia', y='Total_Accidentes', 
                         title='Promedio de Accidentes por Tipo de Día (Datos Insuficientes para ANOVA)',
                         labels={'Tipo_Dia': 'Tipo de Día', 'Total_Accidentes': 'Promedio de Accidentes'})

        fig.update_xaxes(title='Tipo de Día')
        fig.update_yaxes(title='Promedio de Accidentes')
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        return fig
    
    # ANOVA
    model = ols('Total_Accidentes ~ C(Tipo_Dia)', data=filtered_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    # Datos para el gráfico
    mean_accidents = filtered_df.groupby('Tipo_Dia', observed=True)['Total_Accidentes'].mean().reset_index()
    tipo_dia_order = ['Laboral', 'Festivo', 'Fin de Semana Largo']
    mean_accidents['Tipo_Dia'] = pd.Categorical(mean_accidents['Tipo_Dia'], categories=tipo_dia_order, ordered=True)
    mean_accidents = mean_accidents.sort_values('Tipo_Dia')
    
    # Gráfico de puntos con resultados de ANOVA
    fig = px.scatter(mean_accidents, x='Tipo_Dia', y='Total_Accidentes', 
                     title=f'Promedio de Accidentes por Tipo de Día',
                     labels={'Tipo_Dia': 'Tipo de Día', 'Total_Accidentes': 'Promedio de Accidentes'})
    
    fig.update_xaxes(title='Tipo de Día')
    fig.update_yaxes(title='Promedio de Accidentes')
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    
    return fig


@app.callback(
    Output('accidents-by-cause', 'figure'),
    Input('municipio-selector', 'value')
)
def update_accidents_by_cause(selected_municipio):
    if selected_municipio is None:
        return go.Figure()

    filtered_data = df[(df['causaacci'] != 'Certificado cero') & (df['nom_municipio'] == selected_municipio)]
    
    if filtered_data.empty:
        return go.Figure()

    cause_data = filtered_data.groupby('causaacci').size().reset_index(name='accidents')
    cause_data['percentage'] = (cause_data['accidents'] / cause_data['accidents'].sum() * 100).round(2)
    cause_data_encoded = pd.get_dummies(cause_data['causaacci'], drop_first=True)
    X = cause_data_encoded
    y = cause_data['accidents']
    
    if X.isnull().values.any() or y.isnull().values.any():
        raise ValueError("Los datos contienen valores nulos. Por favor, maneje los valores faltantes antes de continuar.")
    
    X = X.astype(float)
    y = y.astype(float)
    
    if len(X) < 2:
        # Si no hay suficientes datos, solo mostrar la gráfica de barras horizontales
        fig = px.bar(cause_data, x='accidents', y='causaacci', orientation='h', 
                     title='Número de Accidentes por Causa (Datos Insuficientes para ANOVA)')
        fig.update_layout(
            title='Número de Accidentes por Causa',
            xaxis_title='Número de Accidentes',
            yaxis_title='Causa del Accidente'
        )
        fig.update_traces(
            text=cause_data['percentage'].astype(str) + '%',
            textposition='outside'
        )
        return fig
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    print(model.summary())
    
    cause_data['predicted'] = model.predict(sm.add_constant(X))
    cause_data['predicted_percentage'] = (cause_data['predicted'] / cause_data['predicted'].sum() * 100).round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cause_data['accidents'],
        y=cause_data['causaacci'],
        orientation='h',
        name='Accidents',
        text=cause_data['percentage'].astype(str) + '%',
        textposition='outside'
    ))

   

    fig.update_layout(
        title='Número de Accidentes por Causa y Predicciones del Modelo',
        xaxis_title='Número de Accidentes',
        yaxis_title='Causa del Accidente',
        barmode='group'
    )
    
    return fig






@app.callback(
    Output('accidents-vehicle', 'figure'),
    Input('municipio-selector', 'value')
)
def update_accidents_vehicle(selected_municipio):
    if selected_municipio is None:
        return go.Figure()

    filtered_df = df[df['nom_municipio'] == selected_municipio]

    if filtered_df.empty:
        return go.Figure()

    vehicle_data = filtered_df[['fecha'] + vehiculo_cols]
    vehicle_data = vehicle_data.melt(id_vars='fecha', var_name='vehiculo', value_name='accidents')
    vehicle_data = vehicle_data.groupby(['fecha', 'vehiculo']).sum().reset_index()
    vehicle_totals = vehicle_data.groupby('vehiculo')['accidents'].sum().reset_index()
    top_vehicle = vehicle_totals.sort_values('accidents', ascending=False).iloc[0]['vehiculo']
    top_vehicle_data = vehicle_data[vehicle_data['vehiculo'] == top_vehicle]

    if top_vehicle_data.empty:
        return go.Figure()

    top_vehicle_data['fecha_ordinal'] = pd.to_datetime(top_vehicle_data['fecha']).map(pd.Timestamp.toordinal)
    X = top_vehicle_data[['fecha_ordinal']]
    y = top_vehicle_data['accidents']
    
    if len(X) < 2:
        # Si no hay suficientes datos, solo mostrar la gráfica sin la tendencia
        fig = go.Figure()
        for vehiculo in vehiculo_cols:
            vehiculo_df = vehicle_data[vehicle_data['vehiculo'] == vehiculo]
            fig.add_trace(go.Scatter(x=vehiculo_df['fecha'], y=vehiculo_df['accidents'],
                                     mode='lines', name=vehiculo))
        fig.update_layout(
            title='Accidentes por Tipo de Vehículo',
            xaxis_title='Fecha',
            yaxis_title='Número de Accidentes'
        )
        return fig
    
    model = LinearRegression()
    model.fit(X, y)
    top_vehicle_data['predictions'] = model.predict(X)
    
    fig = go.Figure()
    for vehiculo in vehiculo_cols:
        vehiculo_df = vehicle_data[vehicle_data['vehiculo'] == vehiculo]
        fig.add_trace(go.Scatter(x=vehiculo_df['fecha'], y=vehiculo_df['accidents'],
                                 mode='lines', name=vehiculo))
    fig.add_trace(go.Scatter(x=top_vehicle_data['fecha'], y=top_vehicle_data['predictions'],
                             mode='lines', name=f'Tendencia {top_vehicle}', line=dict(dash='dash')))
    fig.update_layout(
        title='Accidentes por Tipo de Vehículo',
        xaxis_title='Fecha',
        yaxis_title='Número de Accidentes'
    )
    
    return fig

@app.callback(
    Output('accidents-by-type', 'figure'),
    Input('municipio-selector', 'value')
)
def update_accidents_by_type(selected_municipio):
    if selected_municipio is None:
        return go.Figure()

    filtered_df = df[(df['nom_municipio'] == selected_municipio) & (df['tipaccid'] != 'Certificado cero')]

    if filtered_df.empty:
        return go.Figure()

    type_data = filtered_df.groupby('tipaccid').size().reset_index(name='accidents')
    
    if len(type_data) < 2:
        # Si no hay suficientes datos, solo mostrar la gráfica de barras
        fig = go.Figure()
        fig.add_trace(go.Bar(x=type_data['tipaccid'], y=type_data['accidents'], name='Datos Observados'))
        fig.update_layout(
            title='Número de Accidentes por Tipo de Accidente',
            xaxis_title='Tipo de Accidente',
            yaxis_title='Número de Accidentes'
        )
        return fig
    
    poisson_model = glm('accidents ~ tipaccid', data=type_data, family=sm.families.Poisson()).fit()
    type_data['prediction'] = poisson_model.predict(type_data['tipaccid'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=type_data['tipaccid'], y=type_data['accidents'], name='Datos Observados'))
    fig.add_trace(go.Scatter(x=type_data['tipaccid'], y=type_data['prediction'], mode='lines', name='Predicción'))
    fig.update_layout(
        title='Número de Accidentes por Tipo de Accidente',
        xaxis_title='Tipo de Accidente',
        yaxis_title='Número de Accidentes'
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
