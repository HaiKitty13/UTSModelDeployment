import streamlit as st
import pandas as pd
import pickle

def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        saved = pickle.load(f)
        model = saved["model"]
        scaler = saved["scaler"]
        numeric_columns = saved["numeric_columns"]
        label_mappings = saved["label_mappings"] 
    return model, scaler, numeric_columns, label_mappings

def get_input_data():
    no_of_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=1)
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, max_value=7, value=1)
    no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, max_value=7, value=3)
    type_of_meal_plan = st.selectbox("Meal Plan", options=["Meal Plan 1", "Not Selected", "Meal Plan 2", "Meal Plan 3"])
    required_car_parking_space = st.selectbox("Required Car Parking Space", options=["0.0", "1.0"])
    room_type_reserved = st.selectbox("Room Type Reserved", options=["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    lead_time = st.number_input("Lead Time", min_value=0, max_value=365, value=45)
    arrival_year = st.number_input("Arrival Year", min_value=2022, max_value=2025, value=2023)
    arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=7)
    arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=15)
    market_segment_type = st.selectbox("Market Segment Type", options=["Offline", "Online", "Corporate", "Complementary", "Aviation"])
    repeated_guest = st.number_input("Repeated Guest", min_value=0, max_value=1, value=0)
    no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, max_value=10, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, max_value=10, value=0)
    avg_price_per_room = st.number_input("Average Price per Room", min_value=1, max_value=1000, value=100)
    no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=10, value=1)

    data = pd.DataFrame([{
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }])

    return data

def model_description():
    st.header("Model Description")
    st.markdown("""
        ### Hotel Booking Status Prediction Model
        This model is built using **XGBoost** (Extreme Gradient Boosting) algorithm. 
        It predicts the booking status (whether the booking is confirmed or canceled) based on various input features, 
        such as the number of adults, type of meal plan, room type reserved, and other features related to the booking.

        #### Model Features:
        - **Number of Adults**: Number of adults staying in the hotel
        - **Number of Children**: Number of children staying in the hotel
        - **Number of Weekend Nights**: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
        - **Number of Week Nights**: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
        - **Type of Meal Plan**: Type of meal plan booked by the customer
        - **Required Car Parking Space**: Whether the customer requires a car parking space (0 - No, 1 - Yes)
        - **Room Type Reserved**: Type of room reserved by the customer. Values are encrypted by INN Hotels
        - **Lead Time**: Number of days between the booking date and arrival date
        - **Arrival Year**: Year of arrival date
        - **Arrival Month**: Month of arrival date
        - **Arrival Date**: Day of arrival date
        - **Market Segment Type**: Market segment designation
        - **Repeated Guest**: Whether the customer is a repeated guest (0 - No, 1 - Yes)
        - **Number of Previous Cancellations**: Number of previous bookings canceled by the customer prior to the current booking
        - **Number of Previous Bookings Not Canceled**: Number of previous bookings not canceled by the customer prior to the current booking
        - **Average Price Per Room**: Average price per day of the booking; prices of the rooms are dynamic (in euros)
        - **Number of Special Requests**: Total number of special requests made by the customer (e.g. high floor, view from the room, etc.)
        - **Booking Status**: Target variable indicating if the booking was canceled or not

        #### Performance:
        - **Accuracy**: 89%
        - **Model Type**: XGBoost Classifier
    """)

def main():
    st.set_page_config(page_title="Hotel Booking Prediction", page_icon="🏨", layout="wide")

    st.markdown(
        """
        <style>
        .main {
            background-color: #FAD0C9;
            color: #4E4D4B;
        }
        .stButton>button {
            background-color: #F2A7B8;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #FFE1E6;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #3a3a3a;
            border-radius: 25px;
            padding: 5px;
            width: 100%;
            display: flex;
            justify-content: center;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 25px;
            padding: 10px 20px;
            color: white;
            width: 180px;
            text-align: center;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #F2A7B8 !important;
            color: black !important;
        }
        
        h1 {
            text-align: center;
            color: #4E4D4B;
            margin-bottom: 25px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Hotel Booking Status Prediction")
    st.markdown("<p style='text-align: center; color: #4E4D4B; margin-top: -10px; margin-bottom: 25px;'>Created by Ni Komang Gayatri Kusuma Wardhani    </p>", unsafe_allow_html=True)
    
    tab_prediction, tab_description = st.tabs(["Prediction", "Model Description"])

    with tab_prediction:
        st.header("Make Predictions")
        model, scaler, numeric_columns, label_mappings = load_model()
        data_baru = get_input_data()

        if st.button("Predict Booking Status"):
            for col in data_baru.columns:
                if col in label_mappings:
                    data_baru[col] = data_baru[col].map(label_mappings[col]).fillna(data_baru[col])

            data_baru[numeric_columns] = scaler.transform(data_baru[numeric_columns])

            prediction = model.predict(data_baru)
            
            if prediction[0] == 1:
                predicted_class = "Not Canceled"
            else:
                predicted_class = "Canceled"
            
            st.write(f"Predicted Booking Status: {predicted_class}")

    with tab_description:
        model_description()

if __name__ == "__main__":
    main()
