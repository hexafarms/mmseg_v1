from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File, Query
from predict import segment_api as segment
from fastapi.middleware.cors import CORSMiddleware
from compute_area import *
import os
import io
import re
from PIL import Image
from typing import Union

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
    return {"Hexa Farms'": "MMsegmentation V4"}

@app.post("/segment")
async def create_upload_file(file: UploadFile = File(...), version: Union[str, None] = Query(default=None)):
    file_location = f"fast_api/input/{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    new_version = 0
    versions = [os.path.basename(x[0]) for x in os.walk("/weights")][1:] # exclude the parent path
    
    if version is None:
        "Find the best version if not given"

        if len(versions) == 0:
            NameError("No proper version inside weight folder!")

        for v in versions:
            version = int(re.search('v(.*)', v).group(1))
            if  version > new_version:
                new_version = version 

    else:
        new_version = version[1:]

    ############################ Best Model ###############################
    config_file = f"/weights/v{str(new_version)}/config.py"
    checkpoint_file = f"/weights/v{str(new_version)}/weights.pth"
    #######################################################################

    input_dir = f"fast_api/input/{file.filename}"

    mask, pallete = segment(config_file, checkpoint_file, input_dir) 
    area = compute_area(mask[0], thres=30)
    result = {f'Leaf area of {file.filename} in pixel count'.replace(' ', '_'): f'{int(area)}'}

    bytes_image = io.BytesIO()
    im = Image.fromarray(pallete)
    im.save(bytes_image, format="PNG")

    return Response(content=bytes_image.getvalue(), headers=result, media_type=("image/jpeg"or"image/png"or"image/jpg"))
