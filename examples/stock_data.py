import sys

import pandas as pd

import matplotlib.pyplot as plt

import streamlit as st

import yfinance as yf

from IPython import embed


def plot_timeseries(df, suffix):
    plt.figure()
    df["Close"].plot(style='r', label="closing")
    df['close_ewma'].plot(style='b-.', label="EWMA")
    plt.legend(fontsize=15)
    plt.ylabel("stock prize", fontsize=15)
    plt.tight_layout()
    plt.savefig('plots/ts_{}.pdf'.format(suffix))


def main(args):
    st_MSFT = yf.Ticker("MSFT")

    # get all key value pairs that are available
    for key, value in st_MSFT.info.items():
        print(key, ": ", value)

    stock_list = []
    # FANGMAN
    for stock in ["META", "AMZN", "NFLX", "GOOG", "MSFT", "AAPL", "NVDA"]:
        df = yf.Ticker(stock).history(period="6mo")
        df["close_ewma"] = df["Close"].ewm(alpha=0.1, ignore_na=True).mean()
        plot_timeseries(df, stock)
        stock_list.append(df)

    df = pd.concat(stock_list)
    df_summed = df.groupby(['Date'])[['Close', 'close_ewma']].sum().reset_index()
    plot_timeseries(df_summed, "FANGMAN")

    st.title("Stock Prices")
    st.write("Microsoft Stock Closing Price in USD")
    st.line_chart(st_MSFT.history(period="6mo").Close)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
