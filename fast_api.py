from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File
from predict import segment_api as segment
from fastapi.middleware.cors import CORSMiddleware
from compute_area import *
import io
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hexa Farms'": "MMsegmentation V3"}

@app.post("/segment")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"fast_api/input/{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    ############################ Current Best Model ###############################
    config_file = "/weights/v2/config.py"
    checkpoint_file = "/weights/v2/weights.pth"
    ################################################################################

    ############################### Camera Ratio ##################################
    # 1/10 means 1pixel is 10cm^2 
    ratio = (1/13)**2
    ################################################################################

    input_dir = f"fast_api/input/{file.filename}"

    mask, pallete = segment(config_file, checkpoint_file, input_dir) 
    area = compute_area(mask[0], ratio, thres=30)
    result = {f'Leaf area of {file.filename} in cm2'.replace(' ', '_'): f'{int(area)}'}

    bytes_image = io.BytesIO()
    im = Image.fromarray(pallete)
    im.save(bytes_image, format="PNG")

    return Response(content=bytes_image.getvalue(), headers=result, media_type=("image/jpeg"or"image/png"or"image/jpg"))
