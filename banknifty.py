import json
from datetime import datetime, timedelta
from pytz import timezone
from time import sleep
import pandas as pd
import dash
import os
import csv
# from telegram import Bot  # Import the Bot class from the telegram module
# from telegram.error import TelegramError
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import streamlit as st
import plotly.graph_objects as go
from pya3 import *
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import dash_table
import pymongo
from pymongo import MongoClient


# Replace these with your actual MongoDB connection details
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "bank_nifty"
COLLECTION_NAME = "60479CE"

client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Define your AliceBlue user ID and API key
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)

# Print AliceBlue session ID
print(alice.get_session_id())

# Initialize variables for WebSocket communication
lp = 0
socket_opened = False
subscribe_flag = False
subscribe_list = []
unsubscribe_list = []
data_list = []  # List to store the received data
df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data



# File paths for saving data and graph
data_file_path = "banknifty_60479CE.csv"

graph_file_path = "bank_nifty_60479CE.html"

# Check if the data file exists
if os.path.exists(data_file_path):
    # Load existing data from the CSV file
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
else:
    df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data


all_trend_lines = []
trend_line_visibility = []


# Callback functions for WebSocket connection
def socket_open():
    print("Connected")
    global socket_opened
    socket_opened = True
    if subscribe_flag:
        alice.subscribe(subscribe_list)


def socket_close():
    global socket_opened, lp
    socket_opened = False
    lp = 0
    print("Closed")


def socket_error(message):
    global lp
    lp = 0
    print("Error:", message)

consecutive_green_candles = 0
previous_candle_green = False
label_data = []

# Callback function for receiving data from WebSocket
def feed_data(message):
    global lp, subscribe_flag, data_list, consecutive_green_candles
    feed_message = json.loads(message)
    if feed_message["t"] == "ck":
        print("Connection Acknowledgement status: %s (Websocket Connected)" % feed_message["s"])
        subscribe_flag = True
        print("subscribe_flag:", subscribe_flag)
        print("-------------------------------------------------------------------------------")
        pass
    elif feed_message["t"] == "tk":
        print("Token Acknowledgement status: %s" % feed_message)
        print("-------------------------------------------------------------------------------")
        pass
    else:
        print("Feed:", feed_message)
        if 'lp' in feed_message:
            timestamp = datetime.now(timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S.%f')
            feed_message['timestamp'] = timestamp
            lp = feed_message['lp']
            data_list.append(feed_message)  # Append the received data to the list
            # Insert the data into MongoDB
            collection.insert_one(feed_message)

            # Update marking information only for Heikin Ashi candles
            if len(df) >= 2 and df['mark'].iloc[-1] == '' and feed_message['t'] == 'c':
                if (df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] > df['open'].iloc[-1]):
                    if previous_candle_green:  # Check if the previous candle was green
                        consecutive_green_candles += 1
                        if consecutive_green_candles == 2:  # Mark "YES" only on the second consecutive green candle
                            df.at[df.index[-1], 'mark'] = 'YES'
                    else:
                        consecutive_green_candles = 1
                    previous_candle_green = True
                elif (df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] < df['open'].iloc[-1]):
                    consecutive_green_candles = 0
                    previous_candle_green = False
                    df.at[df.index[-1], 'mark'] = 'NO'

        else:
            print("'lp' key not found in feed message.")


# Connect to AliceBlue

# Socket Connection Request
alice.start_websocket(socket_open_callback=socket_open, socket_close_callback=socket_close,
                      socket_error_callback=socket_error, subscription_callback=feed_data, run_in_background=True,
                      market_depth=False)

while not socket_opened:
    pass

# Subscribe to Tata Motors
subscribe_list = [alice.get_instrument_by_token('NFO', 60479)]
alice.subscribe(subscribe_list)
print(datetime.now())
sleep(15)
print(datetime.now())
# def send_telegram_message(bot_token, chat_id, message):
#     try:
#         bot = Bot(token=bot_token)
#         bot.send_message(chat_id=chat_id, text=message, parse_mode='MarkdownV2')
#     except TelegramError as e:
#         print(f"Error sending Telegram message: {e}")

def calculate_heikin_ashi(data):
    ha_open = (data['open'].shift() + data['close'].shift()) / 2
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_high = data[['high', 'open', 'close']].max(axis=1)
    ha_low = data[['low', 'open', 'close']].min(axis=1)

    ha_data = pd.DataFrame({'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close})
    ha_data['open'] = ha_data['open'].combine_first(data['open'].shift())
    ha_data['high'] = ha_data['high'].combine_first(data['high'])
    ha_data['low'] = ha_data['low'].combine_first(data['low'])
    ha_data['close'] = ha_data['close'].combine_first(data['close'])

    # Add the "mark" column based on Heikin Ashi candle conditions
    ha_data['mark'] = ''

    # Initialize variables to keep track of consecutive green candles and previous YES open
    consecutive_green_candles = 0
    prev_yes_open = None
    no_confirmed = True  # Flag to track if "NO" has been confirmed

    label_data = []  # Create an empty list to store label data

    for i in range(1, len(ha_data)):
        if (ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and
            ha_data['close'].iloc[i] > ha_data['open'].iloc[i]):

            consecutive_green_candles += 1
            prev_yes_open = ha_data['open'].iloc[i]  # Update previous "YES" open value

            if consecutive_green_candles == 1:
                ha_data.at[ha_data.index[i], 'mark'] = 'YES'
                label_data.append(('YES', ha_data.index[i], ha_data['open'].iloc[i], None))
               # send_telegram_message(bot_token, chat_id, f"🟢 **YES**: Candle at {ha_data.index[i]}")
            else:
                ha_data.at[ha_data.index[i], 'mark'] = ''

        elif (ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and
            ha_data['close'].iloc[i] < ha_data['open'].iloc[i]):

            # Check if the previous candle was green
            if consecutive_green_candles > 0:
                ha_data.at[ha_data.index[i], 'mark'] = 'NO'
                if prev_yes_open is not None:
                    if no_confirmed:  # Calculate difference only if "NO" is confirmed
                        confirmed_no_closing = ha_data['close'].iloc[i]  # Store confirmed "NO" closing value
                        diff = prev_yes_open - confirmed_no_closing  # Corrected difference calculation
                        label_data.append(('NO', ha_data.index[i], confirmed_no_closing, diff))
                       # send_telegram_message(bot_token, chat_id, f"🔴 **NO**: Candle at {ha_data.index[i]}")

                    else:
                        ha_data.at[ha_data.index[i], 'mark'] = ''
                        print("Warning: NO not confirmed yet, skipping difference calculation")
                else:
                    ha_data.at[ha_data.index[i], 'mark'] = ''
                    print("Warning: prev_yes_open is None, skipping difference calculation")
       
                no_confirmed = True  # "NO" is confirmed
                consecutive_green_candles = 0  # Reset consecutive_green_candles

    # Calculate the difference and add it to the DataFrame
    ha_data['Difference'] = ha_data['open'] - ha_data['close']

    label_csv_filename = 'labels.csv'
    try:
        with open(label_csv_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Label', 'Timestamp', 'Value', 'Difference'])
            csv_writer.writerows(label_data)
        print(f'Labels saved to {label_csv_filename}')
    except Exception as e:
        print(f'Error saving labels: {e}')

    return ha_data


def main():
    st.title('Market Depth Table')

    # Display the data table
    st.dataframe(df)

    # Rerun the script if the user clicks a button
    if st.button('Refresh Data'):
        st.experimental_rerun()

if __name__ == '__main__':
    main()



def calculate_supertrend(data, atr_period=12, factor=2.0, multiplier=3.0):
    data = data.copy()  # Create a copy of the data DataFrame

    close = data['close']
    high = data['high']
    low = data['low']

    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift())
    tr['l-pc'] = abs(low - close.shift())
    tr['tr'] = tr.max(axis=1)

    atr = tr['tr'].rolling(atr_period).mean()

    median_price = (high + low) / 2
    data['upper_band'] = median_price + (multiplier * atr)
    data['lower_band'] = median_price - (multiplier * atr)

    supertrend = pd.Series(index=data.index)
    direction = pd.Series(index=data.index)

    supertrend.iloc[0] = data['upper_band'].iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(data)):
        if close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(data['lower_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(data['upper_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = -1

        # Start uptrend calculation anew whenever a new uptrend begins
        if direction.iloc[i] == 1 and direction.iloc[i - 1] != 1:
            supertrend.iloc[i] = data['lower_band'].iloc[i]

        # Start downtrend calculation anew whenever a new downtrend begins
        if direction.iloc[i] == -1 and direction.iloc[i - 1] != -1:
            supertrend.iloc[i] = data['upper_band'].iloc[i]

    data['supertrend'] = supertrend  # Add the 'supertrend' column to the data DataFrame
    data['direction'] = direction  # Add the 'direction' column to the data DataFrame

    return data[['open', 'high', 'low', 'close', 'supertrend', 'direction', 'lower_band', 'upper_band']]

def calculate_trend_lines(data):
    current_trend = None
    trend_start = None
    trend_lines = []

    for i in range(len(data)):
        current_signal = data.iloc[i]

        if current_trend is None:
            current_trend = current_signal['direction']
            trend_start = current_signal.name

        if current_trend != current_signal['direction']:
            if trend_start is not None:
                trend_data = data.loc[trend_start:data.index[i - 1]]
                if len(trend_data) > 1:
                    trend_lines.append((current_trend, trend_data))

            current_trend = current_signal['direction']
            trend_start = current_signal.name

    # Handle the last trend if it's still ongoing
    if trend_start is not None and trend_start != data.index[-1]:
        trend_data = data.loc[trend_start:]
        if len(trend_data) > 1:
            trend_lines.append((current_trend, trend_data))

    return trend_lines


all_trend_lines = []



# Function to update the graph

def calculate_current_trend_lines(data):
    current_trend = None
    in_trend = False
    trend_start = None
    trend_lines = []
    buy_signals = pd.DataFrame(columns=['supertrend', 'direction'])
    sell_signals = pd.DataFrame(columns=['supertrend', 'direction'])

    for i in range(len(data)):
        current_signal = data.iloc[i]

        if current_trend is None:
            current_trend = current_signal['direction']
            in_trend = True
            trend_start = current_signal.name

        if current_trend != current_signal['direction']:
            if current_signal['direction'] == 1:
                sell_signals = pd.concat([sell_signals, current_signal])
            else:
                buy_signals = pd.concat([buy_signals, current_signal])

            if in_trend:
                trend_data = data.loc[trend_start:data.index[i - 1]]

                if len(trend_data) > 1:
                    trend_lines.append((current_trend, trend_data))

            else:
                if current_signal['direction'] == 1 and current_trend == -1:
                    updated_trend_data = data.loc[trend_start:data.index[i]]
                    updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
                    current_signal['supertrend'] = updated_supertrend_data['supertrend'].iloc[-1]
                    current_signal['direction'] = updated_supertrend_data['direction'].iloc[-1]
                elif current_signal['direction'] == -1 and current_trend == 1:
                    updated_trend_data = data.loc[trend_start:data.index[i]]
                    updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
                    current_signal['supertrend'] = updated_supertrend_data['supertrend'].iloc[-1]
                    current_signal['direction'] = updated_supertrend_data['direction'].iloc[-1]

            current_trend = current_signal['direction']
            in_trend = False

        if not in_trend:
            if current_trend == 1:
                if not np.isnan(current_signal['upper_band']):
                    trend_start = current_signal.name
                    in_trend = True
            else:
                if not np.isnan(current_signal['lower_band']):
                    trend_start = current_signal.name
                    in_trend = True

    if in_trend:
        trend_data = data.loc[trend_start:]



        if len(trend_data) > 1:
            first_high = trend_data['high'].iloc[0]
            last_close = trend_data['close'].iloc[-1]

    # Handle the continuation of uptrend without a change in direction
    if len(trend_lines) > 0 and data.index[-1] not in trend_lines[-1][1].index and trend_lines[-1][0] == 1:
        last_trend_type, last_trend_data = trend_lines[-1]
        continuation_data = data.loc[data.index > last_trend_data.index[-1]]
        if len(continuation_data) > 1:
            updated_trend_data = pd.concat([last_trend_data.iloc[:-1], continuation_data])
            updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
            continuation_data['supertrend'] = updated_supertrend_data['supertrend'].values[-len(continuation_data):]
            continuation_data['direction'] = updated_supertrend_data['direction'].values[-len(continuation_data):]
            trend_lines[-1] = (last_trend_type, updated_trend_data)

    return trend_lines, buy_signals, sell_signals


# Function to update the graph
def update_graph(n, interval, chart_type):
    global df, data_list, all_trend_lines

    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)

    # Convert 'timestamp' column to datetime and set it as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.set_index('timestamp', inplace=True)

    # Check if there is new data
    if len(data_list) > 0:
        new_df = pd.DataFrame(data_list)
        new_df['lp'] = pd.to_numeric(new_df['lp'], errors='coerce')
        new_df = new_df.dropna(subset=['lp'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        new_df.set_index('timestamp', inplace=True)
        df = pd.concat([df, new_df])

        df.to_csv(data_file_path)
        data_list = []

    df["lp"] = pd.to_numeric(df["lp"], errors="coerce")
    trading_start_time = pd.Timestamp(df.index[0].date()) + pd.Timedelta(hours=9)
    trading_end_time = pd.Timestamp(df.index[0].date()) + pd.Timedelta(hours=23)
    trading_hours_mask = (df.index.time >= trading_start_time.time()) & (df.index.time <= trading_end_time.time())
    df = df[trading_hours_mask]


    # Resample the data for the desired interval
    resampled_data = df["lp"].resample(f'{interval}T').ohlc()
    resampled_data = resampled_data.dropna()

    # Create a datetime index for the x-axis, starting from the first data point and ending at the last data point
    x = pd.date_range(start=df.index[0], end=df.index[-1], freq=f'{interval}T')

    # Plot the data using plotly
    fig = go.Figure(data=[go.Candlestick(x=x,
                open=resampled_data['open'],
                high=resampled_data['high'],
                low=resampled_data['low'],
                close=resampled_data['close'])])

    # Set x-axis label to show only the time
    fig.update_xaxes(type='category', tickformat='%H:%M')

    # Update the layout and display the figure
    fig.update_layout(title=f'Real-Time {chart_type} Chart',
                      xaxis_title='Time',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      yaxis2=dict(overlaying='y', side='left', showgrid=False),
                      template='plotly_dark')


    if chart_type == 'heikin_ashi':
        resampled_data = calculate_heikin_ashi(resampled_data)

        fig = go.Figure()

        # Add Heikin Ashi candlesticks to the figure
        last_timestamp = None  # To track the last timestamp of previous day's data


        # Add Heikin Ashi candlesticks to the figure
        if len(resampled_data) > 0:
            time_difference = (resampled_data.index[0] - resampled_data.index[-1]).total_seconds()
        else:
            time_difference = 0

        # Add Heikin Ashi candlesticks to the figure
        for i, row in enumerate(resampled_data.iterrows()):
            timestamp, candle = row
            candle_color = 'green' if candle['close'] > candle['open'] else 'red'

            # Adjust timestamp by adding the time difference
            timestamp += pd.Timedelta(seconds=time_difference)

            fig.add_trace(go.Candlestick(x=[timestamp],
                                        open=[candle['open']],
                                        high=[candle['high']],
                                        low=[candle['low']],
                                        close=[candle['close']],
                                        increasing_line_color=candle_color,
                                        decreasing_line_color=candle_color,
                                        name=f'Candle {i + 1}'))

            last_timestamp = timestamp
            # Add "yes" or "no" label above the candle
            label_y = None
            label_text = None
            if candle['mark'] == 'YES':
                label_y = candle['high'] + 5  # Adjust this value for proper positioning
                label_text = 'Yes'
            elif candle['mark'] == 'NO':
                label_y = candle['low'] - 15  # Adjust this value for proper positioning
                label_text = 'No'

            if label_y is not None:
                fig.add_annotation(
                    go.layout.Annotation(
                        x=timestamp,
                        y=label_y,
                        text=label_text,
                        showarrow=False,
                        font=dict(color='black', size=12),
                    )
                )

    # Calculate the Supertrend and get the direction from the result
    supertrend_data = calculate_supertrend(resampled_data, factor=2.0)  # Use the new factor parameter
    resampled_data = supertrend_data  # Update resampled_data with the DataFrame returned from calculate_supertrend

    # Add 'volume' column with default value if it doesn't exist in resampled_data
    if 'volume' not in resampled_data:
        resampled_data['volume'] = 0

    # Create a new figure (initialize or update existing figure)
    if 'fig' not in globals():
        fig = plot_candlestick(resampled_data)
        all_trend_lines = []  # Initialize the list of trend lines for the new figure
    else:
        fig = go.Figure()
        all_trend_lines = []


    # Calculate the current trend lines using the updated Supertrend data
    trend_lines, buy_signals, sell_signals = calculate_current_trend_lines(resampled_data)
    trend_lines = calculate_trend_lines(resampled_data)

    for i, (trend_type, trend_data) in enumerate(trend_lines):
        color = 'green' if trend_type == 1 else 'red'
        trend_trace = go.Scatter(
            x=trend_data.index,
            y=trend_data['supertrend'],
            mode='lines',
            name=f'{"Uptrend" if trend_type == 1 else "Downtrend"} Line',
            line=dict(color=color, width=2),
        )

        fig.add_trace(trend_trace)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    # Initialize trend_start and current_trend
    trend_start = None
    current_trend = None

    # Add the sell signals above the candlesticks
    fig.add_trace(go.Scatter(x=sell_signals.index,
                             y=sell_signals['supertrend'],
                             mode='markers',
                             name='Sell Signal',
                             marker=dict(color='green', symbol='triangle-up', size=10)))

    # Add the buy signals below the candlesticks
    fig.add_trace(go.Scatter(x=buy_signals.index,
                             y=buy_signals['supertrend'],
                             mode='markers',
                             name='Buy Signal',
                             marker=dict(color='red', symbol='triangle-down', size=10)))

    fig.write_html(graph_file_path)

    return fig, resampled_data.to_dict('records')

# Function to plot candlestick graph with custom colors
def plot_candlestick(data):
    fig = go.Figure(data=[
        go.Candlestick(x=data.index,
                       open=data['open'],
                       high=data['high'],
                       low=data['low'],
                       close=data['close'],
                       increasing_line_color='green',  # Customize colors here
                       decreasing_line_color='red',   # Customize colors here
                       line=dict(width=1))
    ])



    # Customizing the layout of the graph
    fig.update_layout(
        title="Live Candlestick Graph",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14),
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
    )


    # Add secondary y-axis for price
    fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False))

    return fig
trend_line_visibility = [False] * len(all_trend_lines)

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2"
            "font-family: 'Qwitcher Grypen', cursive;"
        ),
        "rel": "stylesheet",
    },
]
# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets= external_stylesheets)

# MongoDB setup
client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

app.layout = html.Div([
    dcc.Interval(id='graph-update-interval', interval=5000, n_intervals=0),
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Tab 1', value='tab-1'),
        dcc.Tab(label='Tab 2', value='tab-2'),
    ]),

    html.Div(id='tabs-content')
])

# Define the callback to display the appropriate page content based on the URL pathname
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page-2':
        # Fetch data from MongoDB
        data = collection.find({}, {'_id': 0}).sort('timestamp')
        df = pd.DataFrame(data)

        # Convert 'timestamp' column to datetime and set it as the index
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        df.set_index('timestamp', inplace=True)

        # Create and return the content for the second page (Market Depth Table)
        return html.Div([
            html.H3('Market Depth Table'),
            dash_table.DataTable(
                id='data-table',
                columns=[{'name': col, 'id': col} for col in df.columns],
                data=df.to_dict('records'),
                style_table={'height': '1000px', 'overflowY': 'auto'}
            ),
            dcc.Interval(id='table-update-interval', interval=1000, n_intervals=0)
        ])
    else:
        # Default to the first page (Candlestick Chart)
        return html.Div([
            dcc.Graph(id='live-candlestick-graph', config={'displayModeBar': True, 'scrollZoom': True}),
            
            dcc.Dropdown(
                id='chart-type-dropdown',
                options=[
                    {'label': 'Normal', 'value': 'normal'},
                    {'label': 'Heikin Ashi', 'value': 'heikin_ashi'},
                ],
                value='normal',
                clearable=False,
                style={'width': '150px'}
            ),
            dcc.Dropdown(
                id='interval-dropdown',
                options=[
                    {'label': '1 Min', 'value': 1},
                    {'label': '3 Min', 'value': 3},
                    {'label': '5 Min', 'value': 5},
                    {'label': '10 Min', 'value': 10},
                    {'label': '30 Min', 'value': 30},
                    {'label': '60 Min', 'value': 60},
                    {'label': '1 Day', 'value': 1440}
                ],
                value=1,
                clearable=False,
                style={'width': '150px'}
            ),
            dcc.Interval(id='graph-update-interval', interval=2000, n_intervals=0),
            html.Button('Show/Hide Trend Lines', id='toggle-trend-lines-button', n_clicks=0),
        ], style={'height': '100vh', 'width': '100vw'})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H2(["BullsEdges"], className='header-title'),
        html.Nav([
            dcc.Link('Candlestick Chart', href='/', className='nav-link'),
            dcc.Link('Market Depth Table', href='/page-2', className='nav-link'),
            # Add more navigation links as needed
        ], className='nav'),
    ], className='header'),
    html.Div([
        html.Div([
            dcc.Graph(id='live-candlestick-graph', config={'displayModeBar': True, 'scrollZoom': True}),
            
            html.Div([
                html.Label('Chart Type:', className='dropdown-label'),
                dcc.Dropdown(
                    id='chart-type-dropdown',
                    options=[
                        {'label': 'Normal', 'value': 'normal'},
                        {'label': 'Heikin Ashi', 'value': 'heikin_ashi'},
                    ],
                    value='normal',
                    clearable=False,
                    className='dropdown'
                ),
            ], className='dropdown-container'),
            
            html.Div([
                html.Label('Interval:', className='dropdown-label'),
                dcc.Dropdown(
                    id='interval-dropdown',
                    options=[
                        {'label': '1 Min', 'value': 1},
                        {'label': '3 Min', 'value': 3},
                        {'label': '5 Min', 'value': 5},
                        {'label': '10 Min', 'value': 10},
                        {'label': '30 Min', 'value': 30},
                        {'label': '60 Min', 'value': 60},
                        {'label': '1 Day', 'value': 1440}
                    ],
                    value=1,
                    clearable=False,
                    className='dropdown'
                ),
            ], className='dropdown-container'),
            dcc.Interval(id='graph-update-interval', interval=2000, n_intervals=0),
            html.Button('Show/Hide Trend Lines', id='toggle-trend-lines-button', n_clicks=0),
        ], className='content-section'),
    ], className='content'),
    html.Div([
        html.P("Your Footer Information", style={'textAlign': 'center'}),
    ], className='footer'),
])

# Layout of the app

visible_trend_lines = []

# Define the callback to update the data for the market depth table on page two
@app.callback(
    Output('live-candlestick-graph', 'figure'),
    [
        Input('interval-dropdown', 'value'),
        Input('chart-type-dropdown', 'value'),
        Input('live-candlestick-graph', 'relayoutData'),
        Input('toggle-trend-lines-button', 'n_clicks'),
        Input('graph-update-interval', 'n_intervals')
    ],
    [
        State('graph-update-interval', 'n_intervals')
    ]
)
def update_graph_callback(interval, chart_type, relayoutData, n_clicks, n, _):
    fig = go.Figure()
    global all_trend_lines, trend_line_visibility
    fig, _ = update_graph(n, interval, chart_type)

    if 'xaxis.range' in relayoutData:
        xaxis_range = relayoutData['xaxis.range']
    else:
        xaxis_range = [df.index[-1] - pd.Timedelta(minutes=60), df.index[-1]]

    filtered_data = df[(df.index >= xaxis_range[0]) & (df.index <= xaxis_range[1])]

    # Toggle visibility of trend lines based on the button click count
    show_trend_lines = n_clicks % 2 == 1

    # for trend_line, is_visible in zip(all_trend_lines, trend_line_visibility):
    #     # Update trend line visibility
    #     trend_line_idx = all_trend_lines.index(trend_line)
    #     fig.update_traces(
    #         visible=show_trend_lines if is_visible else 'legendonly',
    #         selector=dict(name=f'Uptrend Line {trend_line_idx + 1}')
    #     )
    #     trend_line_visibility[trend_line_idx] = show_trend_lines

    # Update layout with range selector
    fig.update_layout(
        xaxis=dict(
            range=xaxis_range,
            type='date',
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            )
        )
    )
    return fig


# Define the callback to update the data for the data table on page two
@app.callback(
    Output('data-table', 'data'),
    [Input('interval-dropdown', 'value')],
    [dash.dependencies.State('graph-update-interval', 'n_intervals')]
)
def update_data_table(interval, n):
    global df, data_list

    # Append new data to DataFrame
    if len(data_list) > 0:
        new_df = pd.DataFrame(data_list)
        new_df['lp'] = pd.to_numeric(new_df['lp'], errors='coerce')
        new_df = new_df.dropna(subset=['lp'])
        new_df = new_df[["timestamp", "lp"]]
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], format='%Y-%m-%d %H:%M:%S.%f')
        new_df.set_index("timestamp", inplace=True)
        df = pd.concat([df, new_df])

        df.to_csv(data_file_path)
        data_list = []

    # Fetch data from MongoDB
    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)

    # Convert 'timestamp' column to datetime and set it as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.set_index('timestamp', inplace=True)

    # Convert DataFrame to dictionary format for DataTable
    data_table_data = df.to_dict('records')

    return data_table_data
# Run the Dash app
if __name__ == '__main__':
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
    # bot_token = '6694416470:AAG6zGvLWz5s8mIiA9-oSaUf5it7ev-ae1g'
    # chat_id = '987950859'  # Replace with your chat ID
    app.run_server(debug=True)
    # ha_data = calculate_heikin_ashi(df, bot_token, chat_id)