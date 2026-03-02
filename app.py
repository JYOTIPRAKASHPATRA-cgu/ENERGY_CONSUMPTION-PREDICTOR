import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Energy AI Predictor",
    page_icon="⚡",
    layout="wide",
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
h1 {
    text-align: center;
    color: white;
}
.stButton>button {
    background-color: #00c6ff;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA & MODEL ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("small_df.csv")
    df["Day"] = df["Day"].astype("category").cat.codes
    return df

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

df = load_data()
model = load_model()

# ------------------ TITLE ------------------
st.markdown("<h1>⚡ Energy Consumption Prediction System</h1>", unsafe_allow_html=True)
st.markdown("### 🔮 Smart AI Based Energy Forecasting Dashboard")

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙ Input Parameters")

temperature = st.sidebar.slider(
    "🌡 Temperature",
    float(df.Temperature.min()),
    float(df.Temperature.max()),
    float(df.Temperature.mean())
)

humidity = st.sidebar.slider(
    "💧 Humidity",
    float(df.Humidity.min()),
    float(df.Humidity.max()),
    float(df.Humidity.mean())
)

windspeed = st.sidebar.slider(
    "🌬 Wind Speed",
    float(df.WindSpeed.min()),
    float(df.WindSpeed.max()),
    float(df.WindSpeed.mean())
)

hour = st.sidebar.slider("⏰ Hour of Day", 0, 23, 12)
day = st.sidebar.slider("📅 Day (Encoded 0-6)", 0, 6, 3)

# ------------------ PREDICTION ------------------
if st.sidebar.button("🚀 Predict Energy Consumption"):

    input_data = np.array([[temperature, humidity, windspeed, hour, day]])
    prediction = model.predict(input_data)

    st.success(f"⚡ Predicted Energy Consumption: {prediction[0]:.2f} Units")

    # Feature Importance
    st.subheader("📊 Feature Importance")

    importance = model.feature_importances_
    features = df.drop("Energy", axis=1).columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="blues"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

# ------------------ DATA VISUALIZATION ------------------
st.subheader("📈 Energy Trends")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(df, x="Temperature", y="Energy",
                      color="Energy",
                      title="Temperature vs Energy")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(df, x="Hour", y="Energy",
                      color="Energy",
                      title="Hour vs Energy")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 Energy Distribution")

fig3 = px.histogram(df, x="Energy", nbins=30,
                    color_discrete_sequence=["#00c6ff"])
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("### 🚀 Developed by Jyotiprakash Patra | Energy AI Project")