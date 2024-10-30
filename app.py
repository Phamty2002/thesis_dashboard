import os
import pandas as pd
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import logging
import ast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment

# Paths to data
folder_path = os.path.join('Data')
output_folder = os.path.join('Result')

# List of stock symbols (ensure this list is comprehensive or dynamically fetched)
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
        logging.info(f"Loaded '{file_name}' successfully.")
        return df
    else:
        logging.warning(f"Could not find '{file_name}'.")
        return pd.DataFrame()

# Load all forecast summaries
forecast_summary_lstm_svr = load_forecast_summary('forecast_summary_lstm_svr.csv')
forecast_summary_lstm = load_forecast_summary('forecast_summary.csv')
forecast_summary_neural_prophet = load_forecast_summary('forecast_summary_neural_prophet.csv')

# Define app layout with Model Selection
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
                {'label': 'Neural Prophet', 'value': 'Neural_Prophet'}
            ],
            value='LSTM',  # Default model
            clearable=False
        )
    ], style={'width': '20%', 'display': 'inline-block'}),
    
    # Controls: Stock Symbol Selection and Date Range Picker
    html.Div([
        html.Div([
            html.Label('Select Stock Symbol'),
            dcc.Dropdown(
                id='stock-dropdown',
                options=[{'label': s, 'value': s} for s in symbols],
                value='NVDA'  # Default stock
            )
        ], style={'width': '25%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Select Date Range'),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed='2014-09-18',  # Adjust as needed
                max_date_allowed='2025-09-18',  # Adjust as needed
                start_date='2014-09-18',
                end_date='2025-09-18'
            )
        ], style={'display': 'inline-block', 'marginLeft': '50px'}),
        
        html.Div([
            html.Label('Show 1-Year Forecast'),
            dcc.Checklist(
                id='forecast-checkbox',
                options=[{'label': 'Include 1-Year Forecast', 'value': 'show_forecast'}],
                value=[],  # Default unchecked
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
    
    # Hidden Div for Error Messages
    html.Div(id='error-message', style={'color': 'red', 'marginTop': '10px'})
])

# Callback to update graph and metrics based on model selection
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
        logging.info(f"[{model_selected}] Selected date range: {start_date.date()} to {end_date.date()}")
        
        # Retrieve forecast data based on selected model
        if model_selected == 'LSTM_SVR':
            forecast_summary_df = forecast_summary_lstm_svr
        elif model_selected == 'LSTM':
            forecast_summary_df = forecast_summary_lstm
        elif model_selected == 'Neural_Prophet':
            forecast_summary_df = forecast_summary_neural_prophet
        else:
            raise ValueError("Invalid model selected.")
        
        forecast_row = forecast_summary_df[forecast_summary_df['Symbol'] == selected_stock]
        
        if forecast_row.empty:
            logging.error(f"[{model_selected}] No forecast data found for {selected_stock}.")
            return {
                'data': [],
                'layout': go.Layout(
                    title=f'{selected_stock} Price Prediction ({model_selected})',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }, None, f"No forecast data found for {selected_stock}."
        
        # Extract and parse forecast data
        actual_prices = forecast_row.iloc[0].get('Actual_Prices', [])
        predicted_prices = forecast_row.iloc[0].get('Predicted_Prices', [])
        future_prices = forecast_row.iloc[0].get('Future_Price_Predictions', [])
        train_prices = forecast_row.iloc[0].get('Train_Prices', [])
        
        logging.info(f"[{model_selected}] Lengths - Train: {len(train_prices)}, Actual: {len(actual_prices)}, Predicted: {len(predicted_prices)}, Future: {len(future_prices)}")
        
        # Load stock data
        stock_data_file = os.path.join(folder_path, f'{selected_stock}.csv')
        if not os.path.exists(stock_data_file):
            logging.error(f"[{model_selected}] Stock data file not found for {selected_stock}.")
            return {
                'data': [],
                'layout': go.Layout(
                    title=f'{selected_stock} Price Prediction ({model_selected})',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }, None, f"Stock data file not found for {selected_stock}."
        
        df_stock = pd.read_csv(stock_data_file)
        if 'Date' not in df_stock.columns or 'Close' not in df_stock.columns:
            logging.error(f"[{model_selected}] Incorrect data format in stock data file for {selected_stock}.")
            return {
                'data': [],
                'layout': go.Layout(
                    title=f'{selected_stock} Price Prediction ({model_selected})',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }, None, f"Incorrect data format in stock data file for {selected_stock}."
        
        # Process stock data
        df_stock['Date'] = pd.to_datetime(df_stock['Date'])
        df_stock_sorted = df_stock.sort_values('Date')
        dates = df_stock_sorted['Date']
        data_length = len(df_stock)
        
        time_step = 60  # As used in your model
        total_samples = data_length - time_step
        
        # Calculate train_size and test_size based on data length
        samples_per_year = 252
        train_size = 8 * samples_per_year
        test_size = 2 * samples_per_year
        
        # Adjust train_size and test_size if total_samples is less
        if train_size + test_size > total_samples:
            train_size = int(total_samples * 0.8)
            test_size = total_samples - train_size
            logging.warning(f"[{model_selected}] Adjusted train_size to {train_size} and test_size to {test_size} due to insufficient data.")
        
        # Ensure that the length of 'Actual_Prices' matches 'test_size'
        if len(actual_prices) != test_size or len(predicted_prices) != test_size:
            logging.error(f"[{model_selected}] Length mismatch between actual/predicted prices and test set for {selected_stock}.")
            return {
                'data': [],
                'layout': go.Layout(
                    title=f'{selected_stock} Price Prediction ({model_selected})',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }, None, f"Length mismatch in data for {selected_stock}."
        
        # Get indices for train and test sets in original data
        train_indices_in_y_data = range(0, train_size)
        test_indices_in_y_data = range(train_size, train_size + test_size)
        indices_train = [i + time_step for i in train_indices_in_y_data]
        indices_test = [i + time_step for i in test_indices_in_y_data]
        
        # Ensure indices do not exceed the length of dates
        indices_train = [i for i in indices_train if i < len(dates)]
        indices_test = [i for i in indices_test if i < len(dates)]
        
        # Dates for train and test sets
        dates_train = dates.iloc[indices_train].reset_index(drop=True)
        dates_test = dates.iloc[indices_test].reset_index(drop=True)
        
        # Create DataFrame for train set
        df_train = pd.DataFrame({
            'Date': dates_train,
            'Train_Price': train_prices
        })
        
        # Create DataFrame for test set
        df_test = pd.DataFrame({
            'Date': dates_test[:len(actual_prices)],
            'Actual_Price': actual_prices,
            'Predicted_Price': predicted_prices
        })
        
        # Determine the split date (end of train)
        if not df_train.empty:
            split_date = df_train['Date'].max()
        else:
            split_date = None
            logging.warning(f"[{model_selected}] Train data is empty for {selected_stock}.")
        
        # Filter df_train and df_test based on 'start_date' and 'end_date'
        df_train_filtered = df_train[(df_train['Date'] >= start_date) & (df_train['Date'] <= end_date)]
        df_test_filtered = df_test[(df_test['Date'] >= start_date) & (df_test['Date'] <= end_date)]
        
        logging.info(f"[{model_selected}] Filtered data - Train: {len(df_train_filtered)}, Test: {len(df_test_filtered)}")
        
        # Initialize data list for plotting
        plot_data = []
        
        # Add Train Prices if available in the selected date range
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
            logging.info(f"[{model_selected}] Added Train Price trace with {len(df_train_filtered)} points.")
        
        # Add Actual Test Prices
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
            logging.info(f"[{model_selected}] Added Actual Test Price trace with {len(df_test_filtered)} points.")
        
        # Add Predicted Test Prices
        if not df_test_filtered.empty:
            plot_data.append(
                go.Scatter(
                    x=df_test_filtered['Date'],
                    y=df_test_filtered['Predicted_Price'],
                    mode='lines',
                    name='Predicted Test Price',
                    line=dict(color='blue')
                )
            )
            logging.info(f"[{model_selected}] Added Predicted Test Price trace with {len(df_test_filtered)} points.")
        
        # Add vertical line to indicate the split between train and test
        if split_date:
            plot_data.append(
                go.Scatter(
                    x=[split_date, split_date],
                    y=[df_stock_sorted['Close'].min(), df_stock_sorted['Close'].max()],
                    mode='lines',
                    name='Train-Test Split',
                    line=dict(color='black', dash='dash')
                )
            )
            logging.info(f"[{model_selected}] Added Train-Test Split line.")
        
        # Add 1-Year Forecast if selected
        if 'show_forecast' in forecast_option and future_prices:
            # Generate future dates
            last_date = df_stock_sorted['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices), freq='D')
        
            if len(future_prices) != len(future_dates):
                logging.error(f"[{model_selected}] Length mismatch between future prices and future dates for {selected_stock}.")
            else:
                df_future = pd.DataFrame({
                    'Date': future_dates,
                    'Forecasted_Price': future_prices
                })
        
                # Filter future forecasts based on date range
                df_future_filtered = df_future[
                    (df_future['Date'] >= start_date) & (df_future['Date'] <= end_date)
                ]
        
                if not df_future_filtered.empty:
                    plot_data.append(
                        go.Scatter(
                            x=df_future_filtered['Date'],
                            y=df_future_filtered['Forecasted_Price'],
                            mode='lines',
                            name='1-Year Forecast',
                            line=dict(dash='dash', color='orange')
                        )
                    )
                    logging.info(f"[{model_selected}] Added 1-Year Forecast trace with {len(df_future_filtered)} points.")
                else:
                    logging.warning(f"[{model_selected}] No forecasted data within the selected date range for {selected_stock}.")
        
        # Create figure
        figure = {
            'data': plot_data,
            'layout': go.Layout(
                title=f'{model_selected} Model Predictions for {selected_stock} ({start_date.date()} to {end_date.date()})',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'},
                hovermode='closest'
            )
        }
        
        # Extract evaluation metrics
        rmse = forecast_row.iloc[0].get('RMSE', None)
        mse = forecast_row.iloc[0].get('MSE', None)
        mape = forecast_row.iloc[0].get('MAPE', None)
        
        logging.info(f"[{model_selected}] Metrics - RMSE: {rmse}, MSE: {mse}, MAPE: {mape}")
        
        # Prepare metrics output
        if rmse is not None and mse is not None and mape is not None:
            metrics_df = pd.DataFrame({
                'Metric': ['RMSE', 'MSE', 'MAPE'],
                'Value': [f"{rmse:.4f}", f"{mse:.4f}", f"{mape:.2%}"]
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
            logging.info(f"[{model_selected}] Added metrics table.")
        else:
            metrics_output = html.Div([
                html.H4('Evaluation Metrics'),
                html.P("Metrics not available.")
            ])
            logging.warning(f"[{model_selected}] Metrics not available.")
        
        return figure, metrics_output, None  # No error message
    except Exception as e:
        logging.error(f"[{model_selected}] An error occurred while updating the graph: {e}")
        # Return empty graph, no metrics, and an error message
        return {
            'data': [],
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction ({model_selected})',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'}
            )
        }, None, f"An error occurred: {e}"

# Callback to handle download based on model selection
@app.callback(
    Output("download-predictions", "data"),
    [Input("download-button", "n_clicks")],
    [State('model-dropdown', 'value'),
     State('stock-dropdown', 'value')],
    prevent_initial_call=True,
)
def download_predictions(n_clicks, model_selected, selected_stock):
    try:
        logging.info(f"[{model_selected}] Download button clicked for stock: {selected_stock}")
        if n_clicks is None:
            logging.info(f"[{model_selected}] No clicks detected.")
            return dash.no_update

        # Retrieve forecast data based on selected model
        if model_selected == 'LSTM_SVR':
            forecast_summary_df = forecast_summary_lstm_svr
        elif model_selected == 'LSTM':
            forecast_summary_df = forecast_summary_lstm
        elif model_selected == 'Neural_Prophet':
            forecast_summary_df = forecast_summary_neural_prophet
        else:
            raise ValueError("Invalid model selected.")
        
        forecast_row = forecast_summary_df[forecast_summary_df['Symbol'] == selected_stock]
        
        if forecast_row.empty:
            logging.error(f"[{model_selected}] No forecast data found for {selected_stock}.")
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
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices), freq='D')
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
