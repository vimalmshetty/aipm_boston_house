import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt #pip install altair
from plotly import graph_objs as go
from plotly import express  as px

st.title("Boston Housing App")

df = pd.read_csv("data//boston.csv")

st.image("data//boston_house.png")

if st.checkbox("Show Data"):
    st.write(df)

if st.checkbox("Show map"):
    val = st.slider("Filter data based on median value", 0, 40)
    fdata = df.loc[df['MEDV']>=val]
    city_data = fdata[['LON', 'LAT', 'MEDV']]
    city_data.columns = ['longitude', 'latitude','Medv']
    st.map(city_data)

graph = st.selectbox("What kind of Graph?", ["Non-Interactive", "Interactive"])
if graph == 'Non-Interactive':
    fig = sns.relplot(
        data = df,
        x = 'LON',
        y = 'LAT',
        kind='scatter',
        size= 'MEDV',
        hue = 'MEDV',
        s =100,
        aspect=3
    )
    st.pyplot(fig)

if graph == 'Interactive':
    fig = go.Figure(px.scatter(df,
                    x = 'LON',
                    y = 'LAT',
                size = 'MEDV').update_traces(mode="markers"))
    st.plotly_chart(fig)


st.header("Prediction")

#Step 1: Create X and Y

#Create Independent and Dependent Variables
X = df[['CRIM','CHAS','NOX', 'RM','AGE', 'DIS']]
y = df['MEDV']

#Step 2: Split the data in training and testing
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)

#Step 3: Fit the model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_mod = lr.fit(X, y)

#Get the input values
v_room = st.number_input("Enter the number of bedrooms:", 0.00, 10.00, step =1.00, value = 3.00)
v_age = st.slider("Enter the age of the property", 0, 100)
if st.checkbox("Next to Charls Rever?"):
    v_ch = 1
else:
    v_ch = 0
v_dist = st.slider("Distance from the office",0.0,15.0,step=0.5)
v_crim = st.number_input("Enter the preferred crime rate:",0.00, 10.00, step = 0.100, value= 3.00)
v_NOX = st.number_input("Enter the NOX value in the neighborhood:",0.00, 1.00, step = 0.010, value= 0.10)

#Creating test data
test_data = pd.DataFrame(
            dict(CRIM = v_crim,
                CHAS = v_ch,
                NOX = v_NOX,
                RM = v_room,
                AGE = v_age,
                DIS = v_dist),
            index=[0]
        )

if st.button("Predict"):
    pred = lr_mod.predict(test_data)
    st.success(f"Your predicted property value is ${round(pred[0]*1000,2)}" )


# st.header("Home Page")
# st.subheader("First subheader")

# st.write("This is a text block. I am trying to create an application for boston housing problem")

# if st.button("press me!"):
#     st.write("You pressed this button")

# name = st.text_input("Please enter your name:")
# st.write("You entered :" + name)

# date_v = st.date_input("date")
# st.time_input("time")

# st.radio("color", ["r",'g', 'b'])
# st.selectbox("color", ["r",'g', 'b'])