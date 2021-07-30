import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_squared_log_error

@st.cache
def prediction(car_df, carwidth, enginesize, hp, drivewheel_fwd, cc_buick ):
	X = car_df.iloc[ : , : -1]
	y = car_df['price']

	X_train, X_test, y_train, y_test  = train_test_split(X, y, random_state = 42, test_size = 0.3)
	
	lin_reg = LinearRegression()
	lin_reg.fit(X_train, y_train)
	mod_score = lin_reg.score(X_train, y_train)
	
	price = lin_reg.predict([[carwidth, enginesize, hp, drivewheel_fwd, cc_buick]])
	price = price[0]

	ytest_predict = lin_reg.predict(X_test)
	test_r2 = r2_score(y_test, ytest_predict)
	test_mae = mean_absolute_error(y_test, ytest_predict)
	test_rmse = np.sqrt(mean_squared_error(y_test, ytest_predict))
	test_msle = mean_squared_log_error(y_test, ytest_predict)
	return price, mod_score, test_r2, test_mae, test_rmse, test_msle

def app(car_df):
	st.markdown("<p style = 'color:red; font-size:25px'> This app uses <b>LinearRegression</b> to predict the price of the car using the features.", unsafe_allow_html = True)
	st.subheader('Select values')
	
	cw  = st.slider("Car Width", float(car_df['carwidth'].min()), float(car_df['carwidth'].max()))
	esize  = st.slider("Engine Size", float(car_df['enginesize'].min()), float(car_df['enginesize'].max()))
	hpower = st.slider("Horse Power", float(car_df['horsepower'].min()), float(car_df['horsepower'].max()))
	d_fwd = st.radio("Is it a forward driving wheel car?", ('Yes', 'No'))
	
	if d_fwd == 'No':
		d_fwd =0
	else:
		d_fwd = 1
	ccb = st.radio("Is the car manufactured by buick?", ('Yes', 'No'))
	if ccb == 'No':
		ccb = 0
	else:
		ccb  = 1
	
	if st.button("Predict"):
		st.subheader("Prediction results.")
		price, mod_score, test_r2, test_mae, test_rmse, test_msle =  prediction(car_df, cw, esize, hpower, d_fwd, ccb)
		st.success(f"Predicted Price of the car is ${int(price)}.")
		st.info(f"""
			Accuracy score of the model is {mod_score:.2f}.
			Squared error = {test_r2:.2f}.
			Mean absolute error = {test_mae:.2f}
			Root MEan squared error = {test_rmse:.2f}
			Mean Squared log error = {test_msle:.2f}
			""")
