import sys
import streamlit as st
import pandas as pd
import joblib
import traceback
from pathlib import Path

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Real Estate Investment Analyzer",
    layout="wide"
)

st.title("Real Estate Investment Analyzer")
st.warning(
    "This app predicts the dataset's rule-based category. "
    "It does not predict actual property returns or ROI."
)

# -----------------------------
# 2. LOAD TRAINED MODEL
# -----------------------------
MODEL_PATH = Path(__file__).resolve().parent / "real_estate_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {type(e).__name__}: {e}")
    st.stop()


# Use the exact feature order saved by the trained model.
if not hasattr(model, "feature_names_in_"):
    st.error(
        "The model does not expose feature_names_in_. "
        "Check the original training code and saved model."
    )
    st.stop()

features = list(model.feature_names_in_)

st.caption(f"Model loaded. Required features: {len(features)}")

# -----------------------------
# 3. PROPERTY INPUT FORM
# -----------------------------
with st.form("property_form"):

    st.subheader("Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.text_input("State", value="Uttarakhand")
        city = st.text_input("City", value="Dehradun")
        locality = st.text_input("Locality", value="Central")

        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "Independent House", "Villa"]
        )

        bhk = st.number_input("BHK", min_value=1, max_value=20, value=2)
        size = st.number_input(
            "Size (SqFt)", min_value=100, value=1200
        )

        year_built = st.number_input(
            "Year Built", min_value=1900, max_value=2026, value=2020
        )

    with col2:
        furnished = st.selectbox(
            "Furnished Status",
            ["Unfurnished", "Semi-Furnished", "Furnished"]
        )

        floor_no = st.number_input(
            "Floor No", min_value=0, value=2
        )

        total_floors = st.number_input(
            "Total Floors", min_value=1, value=5
        )

        nearby_schools = st.number_input(
            "Nearby Schools", min_value=0, value=3
        )

        nearby_hospitals = st.number_input(
            "Nearby Hospitals", min_value=0, value=2
        )

        transport = st.selectbox(
            "Public Transport Accessibility",
            ["High", "Medium", "Low"]
        )

        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])

    with col3:
        facing = st.selectbox(
            "Facing",
            [
                "North", "South", "East", "West",
                "North-East", "North-West",
                "South-East", "South-West"
            ]
        )

        owner_type = st.selectbox(
            "Owner Type", ["Individual", "Builder", "Resale"]
        )

        availability = st.selectbox(
            "Availability Status",
            ["Ready to Move", "Under Construction"]
        )

        amenities = st.multiselect(
            "Amenities",
            ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
        )

    submitted = st.form_submit_button("Analyze Investment")


# -----------------------------
# 4. PREPARE INPUT AND PREDICT
# -----------------------------
if submitted:

    floor_excess = floor_no - total_floors
    floor_issue = int(floor_excess > 0)

    school_score = min(nearby_schools, 10)
    hospital_score = min(nearby_hospitals, 10)

    # Provide all expected fields; select only the model's features below.
    values = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": 2026 - year_built,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Floor_Issue_Flag": floor_issue,
        "Amenities_Standardized": ", ".join(amenities),
        "Floor_Quality_Status": (
            "Inconsistent" if floor_issue else "Valid"
        ),
        "Floor_Excess": floor_excess,
        "Floor_Difference": floor_excess,
        "Floor_Relationship": (
            "Floor_No above Total_Floors"
            if floor_issue
            else "Floor_No within Total_Floors"
        ),
        "Floor_Above_Total": floor_issue,
        "School_Density_Score": school_score,
        "Hospital_Accessibility_Score": hospital_score,
        "Has_Gym": int("Gym" in amenities),
        "Has_Garden": int("Garden" in amenities),
        "Has_Pool": int("Pool" in amenities),
        "Has_Clubhouse": int("Clubhouse" in amenities),
        "Has_Playground": int("Playground" in amenities),
    }

    missing_features = [f for f in features if f not in values]

    if missing_features:
        st.error(
            "The app does not yet collect these model inputs: "
            + ", ".join(missing_features)
        )
    else:
        # Exact column names and order required by the trained model.
        row = pd.DataFrame(
            [[values[f] for f in features]],
            columns=features
        )

        st.subheader("Prediction Input Check")
        st.write(row)

        try:
            prediction = int(model.predict(row)[0])

            st.subheader("Prediction")

            if prediction == 1:
                st.success(
                    "Good Investment — according to the dataset's label rule."
                )
            else:
                st.info(
                    "Not classified as Good Investment "
                    "according to the dataset's label rule."
                )

            st.caption(
                "This classification is not a verified forecast of "
                "rental yield, appreciation, profit, or ROI."
            )

        except Exception as e:
            st.error(
                f"Prediction failed: {type(e).__name__}: {e}"
            )
            st.code(traceback.format_exc())

