from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import joblib

# Load the pre-trained model
model = joblib.load('final_model.pkl')

# Define the FastAPI app
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define the data model
class Item(BaseModel):
    Gender: int
    Age: int
    Occupation: int
    City_Category: int
    Stay_In_Current_City_Years: int
    Marital_Status: int
    Product_Category_1: int
    Product_Category_2: Optional[float] = None
    Product_Category_3: Optional[float] = None

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            Gender: int = Form(...),
            Age: int = Form(...),
            Occupation: int = Form(...),
            City_Category: int = Form(...),
            Stay_In_Current_City_Years: int = Form(...),
            Marital_Status: int = Form(...),
            Product_Category_1: int = Form(...),
            Product_Category_2: Optional[float] = Form(None),
            Product_Category_3: Optional[float] = Form(None)):
    # Convert the form data to a list of features
    features = [[
        Gender, Age, Occupation, City_Category, 
        Stay_In_Current_City_Years, Marital_Status, 
        Product_Category_1, Product_Category_2, 
        Product_Category_3
    ]]
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Render the template with the prediction
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction[0]})
