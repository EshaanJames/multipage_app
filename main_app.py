import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import plot_confusion_matrix
import data
import plots
import predict

@st.cache()
def load_data():
	df = pd.read_csv('car-prices.csv')
	df = df[['enginesize', 'horsepower', 'carwidth', 'drivewheel', 'price']]
	df['drivewheel'] = df['drivewheel'].map({'rwd': 0, 'fwd': 1, '4wd' : 2})
	return df

car_df = load_data()
st.title("Car Price Prediction App")

with st.beta_expander("Watch the following video...", expanded = False):
	st.write("You can learn how to host a streamlit app on heruko.")
	st.video("https://youtu.be/oBA5I__AfmY")

PAGES = {
	"View Data": data,
	"Visualise Data" : plots,
	"Predict" : predict
}	
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", tuple(PAGES.keys()))
page = PAGES[selection]
page.app(car_df)
