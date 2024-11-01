import os
import pandas as pd
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import logging
import ast
from flask_cors import CORS
from flask import Flask
from dash import Dash

server = Flask(__name__)
CORS(server)

@server.after_request
def add_header(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'  # Cho phép embedding qua iframe
    return response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize Dash app
app = Dash(__name__, server=server)
server = app.server  # For deployment

# Paths to data
folder_path = os.path.join('Data')
output_folder = os.path.join('Result')

# List of stock symbols
symbols = ['NVDA', 'INTC', 'PLTR', 'TSLA', 'AAPL', 'BBD', 'T', 'SOFI',
           'WBD', 'SNAP', 'NIO', 'BTG', 'F', 'AAL', 'NOK', 'BAC',
           'CCL', 'ORCL', 'AMD', 'PFE', 'KGC', 'MARA', 'SLB', 'NU',
           'MPW', 'MU', 'LCID', 'NCLH', 'RIG', 'AMZN', 'ABEV', 'U',
           'LUMN', 'AGNC', 'VZ', 'WBA', 'WFC', 'RIVN', 'UPST', 'GRAB',
           'CSCO', 'VALE', 'AVGO', 'PBR', 'GOOGL', 'SMMT', 'GOLD',
           'CMG', 'BCS', 'UAA']

# Load forecast summaries
def load_forecast_summary(file_name):
    forecast_summary_file = os.path.join(output_folder, file_name)
    if os.path.exists(forecast_summary_file):
        df = pd.read_csv(forecast_summary_file)
        list_columns = ['Predicted_Prices', 'Actual_Prices', 'Future_Price_Predictions', 'Train_Prices']
        for col in list_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
                    logging.info(f"Parsed column '{col}' successfully in '{file_name}'.")
                except (ValueError, SyntaxError) as e:
                    logging.error(f"Error parsing column '{col}' in '{file_name}': {e}")
                    df[col] = [[] for _ in range(len(df))]
        logging.info(f"Loaded '{file_name}' successfully. Number of records: {len(df)}.")
        return df
    else:
        logging.warning(f"Could not find '{file_name}'.")
        return pd.DataFrame()

# Load forecast summaries for all models
forecast_summary_lstm_svr = load_forecast_summary('forecast_summary_lstm_svr.csv')
forecast_summary_lstm = load_forecast_summary('forecast_summary.csv')
forecast_summary_prophet = load_forecast_summary('forecast_summary_Prophet.csv')
forecast_summary_neural_prophet = load_forecast_summary('forecast_summary_neural_prophet.csv')

# *** New Addition: Load SVR forecast summary ***
forecast_summary_svr = load_forecast_summary('forecast_summary_svr.csv')

# *** New Addition: Load XGBoost forecast summary ***
forecast_summary_xgboost = load_forecast_summary('forecast_summary_XGBoost.csv')

# Define app layout
app.layout = html.Div([
    html.H1('Stock Price Predictions Dashboard'),
    
    # Model Selection Dropdown
    html.Div([
        html.Label('Select Model:'),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'LSTM', 'value': 'LSTM'},
                {'label': 'LSTM_SVR', 'value': 'LSTM_SVR'},
                {'label': 'Prophet', 'value': 'Prophet'},
                {'label': 'Neural_Prophet', 'value': 'Neural_Prophet'},
                {'label': 'SVR', 'value': 'SVR'},  # *** Added SVR ***
                {'label': 'XGBoost', 'value': 'XGBoost'}  # *** Added XGBoost ***
            ],
            value='LSTM',
            clearable=False
        )
    ], style={'width': '20%', 'display': 'inline-block'}),
    
    # Controls
    html.Div([
        html.Div([
            html.Label('Select Stock Symbol'),
            dcc.Dropdown(
                id='stock-dropdown',
                options=[{'label': s, 'value': s} for s in symbols],
                value='NVDA'
            )
        ], style={'width': '25%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Select Date Range'),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed='2014-09-18',
                max_date_allowed='2025-09-18',
                start_date='2014-09-18',
                end_date='2025-09-18'
            )
        ], style={'display': 'inline-block', 'marginLeft': '50px'}),
        
        html.Div([
            html.Label('Show 1-Year Forecast'),
            dcc.Checklist(
                id='forecast-checkbox',
                options=[{'label': 'Include 1-Year Forecast', 'value': 'show_forecast'}],
                value=[],
                inline=True
            )
        ], style={'marginTop': '20px', 'display': 'inline-block', 'marginLeft': '50px'})
    ], style={'padding': '20px'}),
    
    # Graph
    dcc.Graph(id='price-graph'),
    
    # Evaluation Metrics
    html.Div(id='metrics-output', style={'marginTop': '20px'}),
    
    # Download Button
    html.Div([
        html.Button("Download Predictions", id="download-button"),
        dcc.Download(id="download-predictions")
    ], style={'marginTop': '20px'}),
    
    # Download Info
    html.Div(id='download-info', style={'marginTop': '10px'}),
    
    # Error Messages
    html.Div(id='error-message', style={'color': 'red', 'marginTop': '10px'})
])

@app.callback(
    [Output('price-graph', 'figure'),
     Output('metrics-output', 'children'),
     Output('error-message', 'children')],
    [Input('model-dropdown', 'value'),
     Input('stock-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('forecast-checkbox', 'value')]
)
def update_graph(model_selected, selected_stock, start_date, end_date, forecast_option):
    try:
        logging.info(f"[{model_selected}] Updating graph for stock: {selected_stock}")
        
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Retrieve forecast data based on selected model
        if model_selected == 'LSTM_SVR':
            forecast_summary_df = forecast_summary_lstm_svr
        elif model_selected == 'LSTM':
            forecast_summary_df = forecast_summary_lstm
        elif model_selected == 'Prophet':
            forecast_summary_df = forecast_summary_prophet
        elif model_selected == 'Neural_Prophet':
            forecast_summary_df = forecast_summary_neural_prophet
        elif model_selected == 'SVR':  # *** Handle SVR ***
            forecast_summary_df = forecast_summary_svr
        elif model_selected == 'XGBoost':  # *** Handle XGBoost ***
            forecast_summary_df = forecast_summary_xgboost
        else:
            raise ValueError("Invalid model selected.")
        
        forecast_row = forecast_summary_df[forecast_summary_df['Symbol'] == selected_stock]
        
        if forecast_row.empty:
            return {
                'data': [],
                'layout': go.Layout(
                    title=f'{selected_stock} Price Prediction ({model_selected})',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }, None, f"No forecast data found for {selected_stock}."
        
        # Extract forecast data
        actual_prices = forecast_row.iloc[0].get('Actual_Prices', [])
        predicted_prices = forecast_row.iloc[0].get('Predicted_Prices', [])
        future_prices = forecast_row.iloc[0].get('Future_Price_Predictions', [])
        train_prices = forecast_row.iloc[0].get('Train_Prices', [])
        
        # Load stock data
        stock_data_file = os.path.join(folder_path, f'{selected_stock}.csv')
        if not os.path.exists(stock_data_file):
            return {
                'data': [],
                'layout': go.Layout(
                    title=f'{selected_stock} Price Prediction ({model_selected})',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }, None, f"Stock data file not found for {selected_stock}."
        
        df_stock = pd.read_csv(stock_data_file)
        df_stock['Date'] = pd.to_datetime(df_stock['Date'])
        df_stock_sorted = df_stock.sort_values('Date')
        dates = df_stock_sorted['Date']
        
        # Calculate indices for train and test sets
        time_step = 60
        data_length = len(df_stock)
        total_samples = data_length - time_step
        samples_per_year = 252
        train_size = 8 * samples_per_year
        test_size = 2 * samples_per_year
        
        if train_size + test_size > total_samples:
            train_size = int(total_samples * 0.8)
            test_size = total_samples - train_size
        
        train_indices = range(time_step, time_step + train_size)
        test_indices = range(time_step + train_size, time_step + train_size + test_size)
        
        # Ensure all arrays have the same length
        min_train_length = min(len(train_indices), len(train_prices))
        min_test_length = min(len(test_indices), len(actual_prices), len(predicted_prices))
        
        # Create DataFrames for plotting with aligned lengths
        df_train = pd.DataFrame({
            'Date': dates.iloc[train_indices[:min_train_length]],
            'Train_Price': train_prices[:min_train_length]
        })
        
        df_test = pd.DataFrame({
            'Date': dates.iloc[test_indices[:min_test_length]],
            'Actual_Price': actual_prices[:min_test_length],
            'Predicted_Price': predicted_prices[:min_test_length]
        })
        
        # Filter data based on selected date range
        df_train_filtered = df_train[
            (df_train['Date'] >= start_date) & 
            (df_train['Date'] <= end_date)
        ]
        
        df_test_filtered = df_test[
            (df_test['Date'] >= start_date) & 
            (df_test['Date'] <= end_date)
        ]
        
        # Create plot data
        plot_data = []
        
        # Add train prices
        if not df_train_filtered.empty:
            plot_data.append(
                go.Scatter(
                    x=df_train_filtered['Date'],
                    y=df_train_filtered['Train_Price'],
                    mode='lines',
                    name='Train Price',
                    line=dict(color='red')
                )
            )
        
        # Add actual test prices
        if not df_test_filtered.empty:
            plot_data.append(
                go.Scatter(
                    x=df_test_filtered['Date'],
                    y=df_test_filtered['Actual_Price'],
                    mode='lines',
                    name='Actual Test Price',
                    line=dict(color='green')
                )
            )
        
            # Add predicted test prices
            plot_data.append(
                go.Scatter(
                    x=df_test_filtered['Date'],
                    y=predicted_prices[:min_test_length],
                    mode='lines',
                    name='Predicted Test Price',
                    line=dict(color='blue')
                )
            )
        
        # Add future predictions if selected
        if 'show_forecast' in forecast_option and future_prices:
            last_date = df_stock_sorted['Date'].max()
            # For models like Prophet and Neural_Prophet, frequency might differ
            if model_selected in ['Prophet', 'Neural_Prophet', 'XGBoost']:  # *** Include XGBoost ***
                freq = 'D'  # Daily frequency
            else:
                freq = 'B'  # Business days
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=len(future_prices),
                freq=freq
            )
            
            df_future = pd.DataFrame({
                'Date': future_dates,
                'Future_Price': future_prices
            })
            
            df_future_filtered = df_future[
                (df_future['Date'] >= start_date) & 
                (df_future['Date'] <= end_date)
            ]
            
            if not df_future_filtered.empty:
                plot_data.append(
                    go.Scatter(
                        x=df_future_filtered['Date'],
                        y=df_future_filtered['Future_Price'],
                        mode='lines',
                        name='1-Year Forecast',
                        line=dict(dash='dash', color='orange')
                    )
                )
        
        # Log array lengths for debugging
        logging.info(f"""
        Array lengths:
        Train indices: {len(train_indices)}
        Test indices: {len(test_indices)}
        Train prices: {len(train_prices)}
        Actual prices: {len(actual_prices)}
        Predicted prices: {len(predicted_prices)}
        """)
        
        # Create figure
        figure = {
            'data': plot_data,
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction ({model_selected})',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'},
                hovermode='closest'
            )
        }
        
        # Prepare metrics output
        metrics = forecast_row.iloc[0]
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MSE', 'MAPE'],
            'Value': [
                f"{metrics.get('RMSE', 'N/A'):.4f}" if pd.notnull(metrics.get('RMSE')) else 'N/A',
                f"{metrics.get('MSE', 'N/A'):.4f}" if pd.notnull(metrics.get('MSE')) else 'N/A',
                f"{metrics.get('MAPE', 'N/A'):.2%}" if pd.notnull(metrics.get('MAPE')) else 'N/A'
            ]
        })
        
        metrics_output = html.Div([
            html.H4('Evaluation Metrics'),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in metrics_df.columns],
                data=metrics_df.to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_table={'width': '50%'}
            )
        ])
        
        return figure, metrics_output, None
        
    except Exception as e:
        logging.error(f"[{model_selected}] An error occurred: {e}")
        return {
            'data': [],
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction ({model_selected})',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'}
            )
        }, None, f"An error occurred: {str(e)}"

@app.callback(
    Output("download-predictions", "data"),
    [Input("download-button", "n_clicks")],
    [State('model-dropdown', 'value'),
     State('stock-dropdown', 'value')],
    prevent_initial_call=True,
)
def download_predictions(n_clicks, model_selected, selected_stock):
    try:
        if n_clicks is None:
            return dash.no_update

        # *** Handle SVR and XGBoost in forecast data selection ***
        if model_selected == 'LSTM_SVR':
            forecast_summary_df = forecast_summary_lstm_svr
        elif model_selected == 'LSTM':
            forecast_summary_df = forecast_summary_lstm
        elif model_selected == 'Prophet':
            forecast_summary_df = forecast_summary_prophet
        elif model_selected == 'Neural_Prophet':
            forecast_summary_df = forecast_summary_neural_prophet
        elif model_selected == 'SVR':  # *** Handle SVR ***
            forecast_summary_df = forecast_summary_svr
        elif model_selected == 'XGBoost':  # *** Handle XGBoost ***
            forecast_summary_df = forecast_summary_xgboost
        else:
            raise ValueError("Invalid model selected.")
        
        forecast_row = forecast_summary_df[forecast_summary_df['Symbol'] == selected_stock]
        
        if forecast_row.empty:
            raise ValueError(f"No forecast data found for {selected_stock}")
            return dash.no_update

        # Extract data
        actual_prices = forecast_row.iloc[0].get('Actual_Prices', [])
        predicted_prices = forecast_row.iloc[0].get('Predicted_Prices', [])
        future_prices = forecast_row.iloc[0].get('Future_Price_Predictions', [])
        train_prices = forecast_row.iloc[0].get('Train_Prices', [])
        
        logging.info(f"[{model_selected}] Lengths for download - Train: {len(train_prices)}, Actual: {len(actual_prices)}, Predicted: {len(predicted_prices)}, Future: {len(future_prices)}")
        
        # Load stock data
        stock_data_file = os.path.join(folder_path, f'{selected_stock}.csv')
        if not os.path.exists(stock_data_file):
            logging.error(f"[{model_selected}] Stock data file not found for {selected_stock}.")
            return dash.no_update
        
        df_stock = pd.read_csv(stock_data_file)
        if 'Date' not in df_stock.columns or 'Close' not in df_stock.columns:
            logging.error(f"[{model_selected}] Incorrect data format in stock data file for {selected_stock}.")
            return dash.no_update
        
        # Convert 'Date' to datetime
        df_stock['Date'] = pd.to_datetime(df_stock['Date'])
        dates = df_stock['Date']
        
        # Determine train-test split date based on 8-year training period
        # Assuming 252 trading days per year
        time_step = 60  # As used in your model
        samples_per_year = 252
        train_size = 8 * samples_per_year
        test_size = 2 * samples_per_year
        total_samples = len(dates) - time_step
        
        if train_size + test_size > total_samples:
            train_size = int(total_samples * 0.8)
            test_size = total_samples - train_size
            logging.warning(f"[{model_selected}] Adjusted train_size to {train_size} and test_size to {test_size} due to insufficient data.")
        
        # Split the data
        train_mask = range(0, train_size)
        test_mask = range(train_size, train_size + test_size)
        
        indices_test = [i + time_step for i in test_mask if (i + time_step) < len(dates)]
        dates_test = dates.iloc[indices_test].reset_index(drop=True)
        
        # Align forecast data with test set dates
        actual_prices = actual_prices[:len(dates_test)]
        predicted_prices = predicted_prices[:len(dates_test)]
        
        if len(actual_prices) != len(predicted_prices):
            logging.error(f"[{model_selected}] Length mismatch between actual and predicted prices for {selected_stock}.")
            return dash.no_update
        
        # Prepare DataFrame for download
        df_test_download = pd.DataFrame({
            'Date': dates_test[:len(actual_prices)],
            'Actual_Price': actual_prices,
            'Predicted_Price': predicted_prices
        })
        
        logging.info(f"[{model_selected}] Prepared test set data for download with {len(df_test_download)} rows.")
        
        # Add future predictions if any
        if future_prices:
            last_date = df_stock['Date'].max()
            # For models like Prophet, Neural_Prophet, and XGBoost, frequency might differ
            if model_selected in ['Prophet', 'Neural_Prophet', 'XGBoost']:
                freq = 'D'  # Daily frequency
            else:
                freq = 'B'  # Business days
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices), freq=freq)
            download_future_df = pd.DataFrame({
                'Date': future_dates,
                'Future_Price_Prediction': future_prices
            })
            df_download = pd.concat([df_test_download, download_future_df], ignore_index=True)
            logging.info(f"[{model_selected}] Added future predictions to download data with {len(download_future_df)} rows.")
        else:
            df_download = df_test_download
            logging.info(f"[{model_selected}] No future predictions to add to download data.")
        
        # Convert DataFrame to CSV and send for download
        logging.info(f"[{model_selected}] Sending download data for {selected_stock}.")
        return dcc.send_data_frame(df_download.to_csv, f'{selected_stock}_{model_selected}_Predictions.csv', index=False)
    except Exception as e:
        logging.error(f"[{model_selected}] An error occurred during download: {e}")
        return dash.no_update

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
