import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def app(car_df):
	st.header("Vidualise Data")
	st.set_option("deprecation.showPyplotGlobalUse", False)
	st.subheader("Scatter Plot")
	features_list = st.multiselect("Select the X-axis values.", ('carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick'))
	for i in features_list:
		st.subheader(f"Scatter plot between {i} and price.")
		plt.figure(figsize = (15, 8))
		sns.scatterplot(x = i, y = 'price', data = car_df)
		st.pyplot()
	st.subheader("Visualisation selector")
	plot_type = st.multiselect("Select Plots to be shown.", ("Histogram", "BoxPlot", "Correlation Heatmap"))
	if "Histogram" in plot_type:
		st.subheader("Histogram")
		columns = st.selectbox("Select the column to create the Histogram",('carwidth', 'enginesize', 'horsepower'))
		plt.figure(figsize = (12,6))
		plt.title(f"Historam for {columns}")

		plt.hist(car_df[columns], bins = 'sturges', edgecolor = 'black')
		st.pyplot()
	if "BoxPlot" in plot_type:
		st.subheader("Boxplot")
		columns = st.selectbox("Select the column to create the Boxplot",('carwidth', 'enginesize', 'horsepower'))
		plt.figure(figsize = (12,6))
		plt.title(f"BoxPlot for {columns}")

		sns.boxplot( car_df[columns])
		st.pyplot()
	if "Correlation Heatmap" in plot_type:
		st.subheader("Correlation Heatmap")
		columns = st.selectbox("Select the column to create the Correlation Heatmap",('carwidth', 'enginesize', 'horsepower'))
		plt.figure(figsize = (12,6))
		plt.title(f"Correlation Heatmap for {columns}")
		sns.heatmap(car_df[columns].corr(), annot = True)
		st.pyplot()
		
		
