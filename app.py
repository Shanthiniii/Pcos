from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse  # Ensure this is included
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Mount static files (for future use, like images or styles)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up the template directory
templates = Jinja2Templates(directory="templates")

# Load the trained model (make sure 'pcos_model.pkl' is in the correct location)
model = pickle.load(open("pcos_model.pkl", "rb"))

# Define the input model for the symptom-based test
class PCOSInput(BaseModel):
    hair_growth: int = 0
    skin_darkening: int = 0
    weight_gain: int = 0
    pimples: int = 0
    hair_loss: int = 0
    cycle: str
    fast_food: str
    weight_Kg: float
    bmi: float
    age_yrs: int
    waist_inch: float
    hip_inch: float
    cycle_length_days: int

# Route to serve the homepage with the general test and professional test buttons
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to serve the symptom-based test page
@app.get("/symptomtest", response_class=HTMLResponse)
async def serve_symptom_test(request: Request):
    return templates.TemplateResponse("symptomtest.html", {"request": request})

# Route to handle form submission and PCOS prediction for symptoms
@app.post("/predict_symptom")
async def predict_symptom(data: PCOSInput):
    # Convert dropdowns to numeric values
    cycle_val = 1 if data.cycle == "R" else 0
    fast_food_val = 1 if data.fast_food == "Y" else 0

    # Create the feature vector
    features = [
        data.hair_growth,
        data.skin_darkening,
        data.weight_gain,
        cycle_val,         # Convert "R"/"I" cycle to numeric
        fast_food_val,     # Convert "Y"/"N" fast_food to numeric
        data.pimples,
        data.weight_Kg,
        data.bmi,
        data.waist_inch,
        data.age_yrs,
        data.hair_loss,
        data.hip_inch,
        data.cycle_length_days
    ]

    # Columns used during training the model
    columns = [
        "hair growth(Y/N)", "Skin darkening (Y/N)", "Weight gain(Y/N)", "Cycle(R/I)",
        "Fast food (Y/N)", "Pimples(Y/N)", "Weight (Kg)", "BMI", "Waist(inch)",
        "Age(yrs)", "Hair loss(Y/N)", "Hip(inch)", "Cycle length(days)"
    ]
    
    features_df = pd.DataFrame([features], columns=columns)

    # Make the prediction using the model
    prediction = model.predict(features_df)[0]

    # Return the prediction result
    if prediction == 1:  # Assuming 1 indicates PCOS risk
        return {"prediction": "PCOS", "message": "Please consult a doctor."}
    else:
        return {"prediction": "No PCOS", "message": "You are not showing symptoms of PCOS."}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
