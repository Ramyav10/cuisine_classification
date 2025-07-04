import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

def add_bg():
    st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        height: 100%;
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stApp {
        background-image: url("https://img.freepik.com/premium-photo/sushi-set-wooden-table_961875-29830.jpg?w=360");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        min-height: 100vh;

    
    .block-container {
        background-color: rgba(255,255,255,0.75);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        max-width: 600px;
        margin: auto;
    }
    div.stButton > button {
        background-color: #007bff;  /* Blue */
        color: white;
        border: none;
        padding: 0.5rem 1.25rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
    }
    div.stButton > button:hover {
        background-color: #005ec2;
    }
    a {
        color: #007bff;
        text-decoration: none;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

USER_FILE = "users.json"
def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def user_exists(username):
    return username in load_users()

def validate_user(username, password):
    return load_users().get(username) == password

#  Load model and data
@st.cache_resource
def load_model():
    model = joblib.load("multilabel_model.pkl")
    mlb = joblib.load("multilabel_binarizer.pkl")
    df = pd.read_csv("cleaned_dataset13.csv")
    return model, mlb, df

model, mlb, df = load_model()
cities = sorted(df["City"].dropna().unique())

#  Page  Setup
st.set_page_config(page_title="Cuisine Tagger", layout="centered")
add_bg()

if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

#  Welcome page
if st.session_state.page == "welcome":
    st.title("Automated Cuisine Tagger")
    st.markdown("Welcome! Choose an option to continue:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Register"):
            st.session_state.page = "register"
    with col2:
        if st.button(" Login"):
            st.session_state.page = "login"

# Register Page
elif st.session_state.page == "register" and not st.session_state.logged_in:
    st.title("Register")
    email = st.text_input("Email / Username")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):
        if not email or not password:
            st.warning("All fields are required.")
        elif password != confirm:
            st.error("Passwords do not match.")
        elif user_exists(email):
            st.error("User already exists.")
        else:
            save_user(email, password)
            st.success("Registration successful! Please log in.")
            st.session_state.page = "login"

    if st.button("⬅ Back to Welcome"):
        st.session_state.page = "welcome"

    st.markdown("Already have an account?", unsafe_allow_html=True)
    if st.button("Login here", key="login_link", help="Click to login", type="primary"):
        st.session_state.page = "login"

    if "login" in st.query_params:
        st.session_state.page = "login"

# Login Page
elif st.session_state.page == "login" and not st.session_state.logged_in:
    st.title("Login")
    email = st.text_input("Email / Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if validate_user(email, password):
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.page = "main"
        else:
            st.error("Invalid username or password.")

    if st.button("⬅ Back to Welcome"):
        st.session_state.page = "welcome"

    st.markdown("Don't have an account?", unsafe_allow_html=True)
    if st.button("Register here", key="register_link", help="Click to register", type="primary"):
        st.session_state.page = "register"

#  Prediction Page
elif st.session_state.logged_in:
    st.title("Cuisine Tagger")
    st.markdown("Fill in restaurant details to predict likely cuisines:")

    city = st.selectbox("City", cities)
    price_range = st.slider("Price Range (1 = Low, 4 = High)", 1, 4, 2)
    cost = st.number_input("Average Cost for Two (₹)", value=500)
    votes = st.number_input("Number of Votes", value=100)
    rating = st.slider("Aggregate Rating", 0.0, 5.0, 3.5)
    table_booking = st.radio("Has Table Booking?", ["Yes", "No"], horizontal=True)
    online_delivery = st.radio("Has Online Delivery?", ["Yes", "No"], horizontal=True)
    threshold = st.slider("Confidence Threshold (%)", 0, 100, 50)

    if st.button("Predict Cuisines"):
        city_code = hash(city) % 100
        log_cost = np.log1p(cost)
        log_votes = np.log1p(votes)
        votes_per_rating = votes / (rating + 0.1)
        cost_per_vote = cost / (votes + 1)
        has_table = 1 if table_booking == "Yes" else 0
        has_online = 1 if online_delivery == "Yes" else 0

        input_df = pd.DataFrame([[
            city_code, log_cost, has_table, has_online,
            price_range, rating, log_votes, votes_per_rating, cost_per_vote
        ]], columns=[
            "City_Code", "log_cost", "Has Table booking", "Has Online delivery",
            "Price range", "Aggregate rating", "log_votes", "Votes_per_rating", "Cost_per_vote"
        ])

        probs = np.array([clf.predict_proba(input_df)[0][1] for clf in model.estimators_])
        results = [(mlb.classes_[i], probs[i]) for i in range(len(probs)) if probs[i] >= (threshold / 100)]
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            st.success("🍴 Predicted Cuisines with Confidence:")
            for cuisine, prob in results[:5]:
                st.markdown(f"- **{cuisine}** — `{prob:.0%}` confidence")
        else:
            st.warning("No cuisines passed the confidence threshold.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "welcome"