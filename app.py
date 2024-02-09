import io
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
from pydantic import BaseModel
from moviepy.editor import VideoFileClip
from PIL import ImageDraw, ImageFont

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileRequest(BaseModel):
    file: str
    
custom_model = load_model('custom_vehicle_recognition_model.h5')
custom_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the path to your dataset
dataset_path = 'dataset'
train_data_dir = os.path.join(dataset_path, 'train')

# Extract class labels from the directory structure
class_labels = sorted(os.listdir(train_data_dir))

def process_predictions(predictions, class_labels):
    if len(predictions.shape) == 2 and predictions.shape[0] == 1:
        predictions = predictions[0]

    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[predicted_class_index]

    return predicted_class_label, confidence

def predict_image(img_array):
    img_array = preprocess_input(img_array)
    predictions = custom_model.predict(img_array)
    return predictions

def count_classified_vehicles(predictions, threshold):
    counts = {}
    for i, pred in enumerate(predictions):
        predicted_class, confidence = process_predictions(pred, class_labels)
        if confidence >= threshold:
            counts[predicted_class] = counts.get(predicted_class, 0) + 1
    return counts

def mark_vehicles(image, vehicles):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)

    for i, (vehicle, count) in enumerate(vehicles.items()):
        text = f'{vehicle}: {count}'
        draw.text((10, 10 + 25 * i), text, fill=(255, 0, 0), font=font)

    return image

def process_video_with_marking(video_bytes):
    try:
        video = VideoFileClip(io.BytesIO(video_bytes))
        frame_count = int(video.duration * video.fps)
        frames_with_markings = []

        for i in range(frame_count):
            frame = video.get_frame(i / video.fps)
            img = image.array_to_img(frame)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = predict_image(img_array)
            frame_counts = count_classified_vehicles([predictions], 0.5)

            marked_img = mark_vehicles(img, frame_counts)
            img_bytes = io.BytesIO()
            marked_img.save(img_bytes, format='JPEG')
            frames_with_markings.append(img_bytes.getvalue())

        return frames_with_markings

    except Exception as e:
        print(f'Error processing video {e}')
        raise HTTPException(status_code=500, detail=str(e))

def process_image_with_marking(image_bytes):
    try:
        img = image.load_img(io.BytesIO(image_bytes), target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = predict_image(img_array)
        predicted_class, confidence = process_predictions(predictions, class_labels)

        marked_img = mark_vehicles(img, {predicted_class: confidence})
        img_bytes = io.BytesIO()
        marked_img.save(img_bytes, format='JPEG')
        
        return img_bytes.getvalue()

    except Exception as e:
        print(f'Error processing image {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uploadFile/")
async def create_upload_file(file: UploadFile = File(...)):
    result = process_image_with_marking(await file.read())
    return StreamingResponse(io.BytesIO(result), media_type="image/jpeg")

@app.post("/uploadVideo/")
async def create_upload_video(file: UploadFile = File(...)):
    result = process_video_with_marking(await file.read())
    return JSONResponse(content={"frames_data": result})
