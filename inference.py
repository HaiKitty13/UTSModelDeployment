import pandas as pd
import pickle

class XGBoostInference:
    def __init__(self, model_path, target_column):
        print("Loading model...")
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
            self.model = saved["model"]
            self.scaler = saved["scaler"]
            self.numeric_columns = saved["numeric_columns"]
            self.label_mappings = saved["label_mappings"]
            self.reverse_label_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in self.label_mappings.items()}
        
        self.target_column = target_column

    def preprocess_input(self, input_data: pd.DataFrame):
        input_data = input_data.copy()

        for col, mapping in self.label_mappings.items():
            if col in input_data.columns:
                try:
                    input_data[col] = input_data[col].map(mapping)
                except Exception as e:
                    raise ValueError(f"Error in mapping column '{col}': {e}")
        
        input_data[self.numeric_columns] = self.scaler.transform(input_data[self.numeric_columns])
        return input_data

    def predict(self, input_data: pd.DataFrame):
        processed_data = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_data)

        predicted_class = self.reverse_label_mappings[self.target_column].get(prediction[0], prediction[0])
        return predicted_class

if __name__ == "__main__":
    data_baru = pd.DataFrame([{
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 3,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": "0.0",
        "room_type_reserved": "Room_Type 1",
        "lead_time": 45,
        "arrival_year": 2023,
        "arrival_month": 7,
        "arrival_date": 15,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 100,
        "no_of_special_requests": 1
    }])

    infer = XGBoostInference("xgboost_model.pkl", target_column="booking_status")
    hasil = infer.predict(data_baru)
    print("Prediksi Booking Status:", hasil)
