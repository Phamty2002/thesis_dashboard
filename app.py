import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Import the create_layout functions from each dashboard
from dashboard_LSTM import create_layout as create_lstm_layout
from dashboard_LSTM_SVR import create_layout as create_lstm_svr_layout
# Import other dashboards similarly...

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

# Main layout with navigation
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

# Callback to render the selected dashboard
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/dashboard_LSTM':
        return create_lstm_layout(app)
    elif pathname == '/dashboard_LSTM_SVR':
        return create_lstm_svr_layout(app)
    elif pathname == '/dashboard_NP':
        return create_np_layout(app)
    elif pathname == '/dashboard_Pr':
        return create_pr_layout(app)
    elif pathname == '/dashboard_SVR':
        return create_svr_layout(app)
    elif pathname == '/dashboard_XG':
        return create_xg_layout(app)
    else:
        return html.Div([
            html.H3('Welcome to the Forecast Dashboard! Please select a dashboard from the navigation menu.')
        ])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
