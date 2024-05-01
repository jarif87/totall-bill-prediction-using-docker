import pickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Load the trained model
with open("rf_model_3.pkl", "rb") as file:
    model = pickle.load(file)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request,
                  tip: float = Form(...),
                  sex: int = Form(...),
                  smoker: int = Form(...),
                  day: int = Form(...),
                  time: int = Form(...),
                  size: int = Form(...)):

    features = [tip, sex, smoker, day, time, size]
    prediction = model.predict([features])[0]

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
