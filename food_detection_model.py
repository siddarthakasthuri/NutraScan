import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load food data
food_data = pd.read_excel(r"C:\Users\Admin\Desktop\working1\yoyo.xlsx")
food_calories = food_data.set_index('food')['calories'].to_dict()
food_ingredients = food_data.set_index('food')['ingredients'].to_dict()
# image_path=r"C:\Users\Admin\Pictures\Camera Roll\WhatsApp Image 2024-04-20 at 9.54.28 PM.jpeg"
# Initialize YOLO model
model = YOLO(r"D:\Food_Images.v1i.yolov8\runs\detect\train\weights\last.pt")
def detect_food(image_path):
    # Predict using YOLO model
    results = model.predict(image_path)
    for result in results:
        if result.boxes:
            box = result.boxes[0]
            class_id = int(box.cls)
            object_name = model.names[class_id]
            return object_name

def calculate_food_size(image_path):
    # Initialize ARUCO detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Read image and detect markers
    image = cv2.imread(image_path) 
    corners, ids, rejectedImgPoints = detector.detectMarkers(image)

    # Calculate size based on marker and object dimensions
    if ids is not None and len(ids) > 0:
        marker_size = 10  # Size of the marker in centimeters (you need to adjust this based on your actual marker size)
        pixel_size = corners[0][0][1][0] - corners[0][0][0][0]  # Assuming the marker is a square, calculate the size of one side in pixels
        cm_per_pixel = marker_size / pixel_size  # Calculate conversion factor from pixels to centimeters

    results = model.predict(image)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1y1x2y2 = box.xywh.cpu().numpy()
            array = np.array(x1y1x2y2)
            nested_array = array[0] 
            x1, y1, x2, y2 = nested_array
            width = x2 - x1
            length = y2- y1
    object_size = abs(width * length * cm_per_pixel)
    return object_size

def get_calorie_count(image_path):
    food_name = detect_food(image_path)
    if food_name is None:
        return 'Unknown'
    else:
        calorie = food_calories.get(food_name.lower(), 'Unknown')
        calorie_count = calculate_food_size(image_path) * calorie
        return calorie_count

def get_ingredients(image_path):
    food_name = detect_food(image_path)
    if food_name is None:
        return 'Unknown'
    else:
        ing = food_ingredients.get(food_name.lower(), 'No ingredients found')
        ingredients_list = ing.split(", ")
        return '\n'.join(ingredients_list)

# calculate_food_size(image_path)
