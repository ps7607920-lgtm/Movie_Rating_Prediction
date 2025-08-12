import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

data = pd.read_csv("IMDb Movies India.csv.zip", encoding="latin")

data["Duration"] = data["Duration"].str.replace(" min", "").astype(float)
data["Year"] = data["Year"].str.strip("()").astype(float)
data["Votes"] = data["Votes"].str.replace(",", "")
data["Votes"] = data["Votes"].str.replace("$5.16M", "516", regex=False)
data["Votes"] = data["Votes"].astype(float)

data["Duration"].fillna(data["Duration"].median(), inplace=True)
data["Rating"].fillna(data["Rating"].mean(), inplace=True)
data["Votes"].fillna(data["Votes"].median(), inplace=True)
data.dropna(subset=["Genre", "Actor 1", "Year", "Director", "Actor 2", "Actor 3"], inplace=True)

data["Genre"] = data["Genre"].str.split(",").explode("Genre").reset_index(drop=True)

data["Genre"] = data.groupby("Genre")["Rating"].transform("mean")
data["Director"] = data.groupby("Director")["Rating"].transform("mean")
data["Actor 1"] = data.groupby("Actor 1")["Rating"].transform("mean")
data["Actor 2"] = data.groupby("Actor 2")["Rating"].transform("mean")
data["Actor 3"] = data.groupby("Actor 3")["Rating"].transform("mean")
data["Name"] = data.groupby("Name")["Rating"].transform("mean")

data.drop("Duration", axis=1, inplace=True)
features = data.drop("Rating", axis=1)
target = data["Rating"]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

st.title("🎬 IMDb India Movie Rating Predictor")
st.write("Enter movie details to predict the IMDb rating.")

genre = st.text_input("Genre")
director = st.text_input("Director")
actor1 = st.text_input("Actor 1")
actor2 = st.text_input("Actor 2")
actor3 = st.text_input("Actor 3")

if st.button("Predict Rating"):
    genre_val = data.groupby("Genre")["Genre"].mean().get(genre, data["Genre"].mean())
    director_val = data.groupby("Director")["Director"].mean().get(director, data["Director"].mean())
    actor1_val = data.groupby("Actor 1")["Actor 1"].mean().get(actor1, data["Actor 1"].mean())
    actor2_val = data.groupby("Actor 2")["Actor 2"].mean().get(actor2, data["Actor 2"].mean())
    actor3_val = data.groupby("Actor 3")["Actor 3"].mean().get(actor3, data["Actor 3"].mean())

    input_df = pd.DataFrame([[data["Name"].mean(), data["Year"].mean(), genre_val, data["Votes"].mean(),
                              director_val, actor1_val, actor2_val, actor3_val]],
                            columns=["Name", "Year", "Genre", "Votes", "Director", "Actor 1", "Actor 2", "Actor 3"])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted IMDb Rating: {prediction:.2f}")
