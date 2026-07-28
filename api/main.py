from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from perception.inference.pipeline import PerceptionPipeline

app = FastAPI(title="Autonomous Driving Perception API", version="0.2.0")

# Load pipeline
print("Loading perception pipeline...")
pipeline = PerceptionPipeline()
print("Pipeline is ready!")

@app.get("/")
def root():
    return {"message": "Autonomous Driving Perception API is running", "status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    result = pipeline.detector.detect(image)
    boxes = pipeline.detector.get_boxes(result)
    
    return JSONResponse({
        "objects": boxes,
        "total_objects": len(boxes)
    })

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    _, classes_found = pipeline.segmentor.segment(image)
    
    return JSONResponse({
        "classes_found": classes_found,
        "num_classes": len(classes_found)
    })

@app.post("/pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    result = pipeline.run(image)
    return JSONResponse(result)