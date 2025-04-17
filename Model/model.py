import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

class XGBoostTrainer:
    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.target_column = target_column
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.label_mappings = {}

    def load_data(self):
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        print("Missing values per column:\n", df.isnull().sum())

        self.df = df.dropna()
        self.df = self.df.drop(["Booking_ID"], axis=1)

        self.numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = self.df.select_dtypes(exclude=["number"]).columns.tolist()

    def encode_data(self):
        for col in self.cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            self.label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    def defineXnY(self):
        self.X = self.df.drop(self.target_column, axis=1)
        self.y = self.df[self.target_column]

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def scaling(self):
        self.scaler = StandardScaler()
        self.X_train[self.numeric_cols] = self.scaler.fit_transform(self.X_train[self.numeric_cols])
        self.X_test[self.numeric_cols] = self.scaler.transform(self.X_test[self.numeric_cols])

    def train_model(self):
        print("Training XGBoost model...")
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        print("Evaluating model...")
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        print(f"Accuracy: {acc}")
        print("Classification Report:\n", report)

    def save_model(self, filename="xgboost_model.pkl"):
        print(f"Saving model to {filename}...")
        with open(filename, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "numeric_columns": self.numeric_cols,
                "label_encoders": self.label_encoders,
                "label_mappings": self.label_mappings
            }, f)

if __name__ == "__main__":
    trainer = XGBoostTrainer(data_path="Model/Dataset_B_hotel.csv", target_column="booking_status")
    trainer.load_data()
    trainer.encode_data()
    trainer.defineXnY()
    trainer.split_data()
    trainer.scaling()
    trainer.train_model()
    trainer.evaluate()
    trainer.save_model()
