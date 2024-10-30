# app.py

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Import các file dashboard
import dashboard_LSTM
import dashboard_LSTM_SVR
import dashboard_NP
import dashboard_Pr
import dashboard_SVR
import dashboard_XG

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

# Layout chính với menu điều hướng
app.layout = html.Div([
    dbc.NavbarSimple(
        brand="Forecast Dashboards",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("LSTM Dashboard", href="/dashboard_LSTM")),
            dbc.NavItem(dbc.NavLink("LSTM_SVR Dashboard", href="/dashboard_LSTM_SVR")),
            dbc.NavItem(dbc.NavLink("Neural Prophet Dashboard", href="/dashboard_NP")),
            dbc.NavItem(dbc.NavLink("Prophet Dashboard", href="/dashboard_Pr")),
            dbc.NavItem(dbc.NavLink("SVR Dashboard", href="/dashboard_SVR")),
            dbc.NavItem(dbc.NavLink("XGBoost Dashboard", href="/dashboard_XG")),
        ],
    ),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Callback để điều hướng và hiển thị dashboard tương ứng
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/dashboard_LSTM':
        return dashboard_LSTM.create_layout(app)
    elif pathname == '/dashboard_LSTM_SVR':
        return dashboard_LSTM_SVR.create_layout(app)
    elif pathname == '/dashboard_NP':
        return dashboard_NP.create_layout(app)
    elif pathname == '/dashboard_Pr':
        return dashboard_Pr.create_layout(app)
    elif pathname == '/dashboard_SVR':
        return dashboard_SVR.create_layout(app)
    elif pathname == '/dashboard_XG':
        return dashboard_XG.create_layout(app)
    else:
        return html.Div([
            html.H3('Welcome to the Forecast Dashboard! Please select a dashboard from the navigation menu.')
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
