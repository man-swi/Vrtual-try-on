# BodySegmentation.py
import numpy as np
# import cv2 # Removed unused import
from PIL import Image, ImageDraw
import insightface
from insightface.app import FaceAnalysis
from transformers import pipeline

print("Initializing FaceAnalysis for body_segmentation (loading only detection and landmarks)...")
face_analyzer = FaceAnalysis(
    name='buffalo_l', # Keep using parts of the buffalo_l bundle
    allowed_modules=['detection', 'landmark_2d_106'], # Specify only these
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
# This will still download all models of buffalo_l if not cached,
# but will only load the specified ones into VRAM via ONNXRuntime.
face_analyzer.prepare(ctx_id=0, det_size=(640, 640)) # prepare is still needed
print("FaceAnalysis (selective modules) initialized.")

print("Initializing segmentation_model for body_segmentation...")
segmentation_model = pipeline(model="mattmdjaga/segformer_b2_clothes")
print("Segmentation_model initialized.")

def mask_face(img_pil_rgb, mask_pil_l): # Expecting PIL RGB image and PIL L mask
    # img_pil_rgb is the image from which face is detected
    # mask_pil_l is the mask to modify
    
    # Insightface expects BGR numpy array
    img_bgr_numpy = np.array(img_pil_rgb)[:, :, ::-1].copy() # Ensure it's BGR

    detected_faces = face_analyzer.get(img_bgr_numpy)

    if not detected_faces:
        return mask_pil_l # Return original mask if no face

    face_bbox = detected_faces[0]['bbox'] 
    face_x1, face_y1, face_x2, face_y2 = map(int, face_bbox)

    face_width = face_x2 - face_x1
    face_height = face_y2 - face_y1
   
    expanded_x1 = face_x1 - face_width * 0.5
    expanded_x2 = face_x2 + face_width * 0.5
    expanded_y1 = face_y1 - face_height * 0.5 # Original paper might have different expansion for y1
    expanded_y2 = face_y2 + face_height * 0.2

    img_width, img_height = mask_pil_l.size
    
    expanded_x1 = max(0, int(expanded_x1))
    expanded_y1 = max(0, int(expanded_y1))
    expanded_x2 = min(img_width -1 , int(expanded_x2))
    expanded_y2 = min(img_height -1, int(expanded_y2))

    if expanded_x1 >= expanded_x2 or expanded_y1 >= expanded_y2: # Check for invalid rectangle
        return mask_pil_l

    face_coords = [(expanded_x1, expanded_y1), (expanded_x2, expanded_y2)]
    
    # Create a drawable version of the mask to modify
    draw = ImageDraw.Draw(mask_pil_l)
    draw.rectangle(face_coords, fill=0) 

    return mask_pil_l

def apply_segmentation(original_img_pil, include_face=True):
    img_pil_rgb = original_img_pil.copy().convert("RGB") 
    
    # Segmentation model expects PIL image
    segments = segmentation_model(img_pil_rgb)

    segment_labels = ["Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]
    
    combined_mask_np = np.zeros((img_pil_rgb.height, img_pil_rgb.width), dtype=np.uint8)

    for s in segments:
        if s['label'] in segment_labels:
            mask_np = np.array(s['mask'].convert('L')) # Ensure mask is L mode before converting to array
            combined_mask_np = np.maximum(combined_mask_np, mask_np) 
    
    combined_mask_pil_l = Image.fromarray(combined_mask_np, mode='L')

    if not include_face:
        combined_mask_pil_l = mask_face(img_pil_rgb, combined_mask_pil_l)

    img_rgba = img_pil_rgb.copy() 
    img_rgba.putalpha(combined_mask_pil_l) 

    return img_rgba, combined_mask_pil_l # Return the alpha image and the L mask

def apply_torso_segmentation(original_img_pil):
    img_pil_rgb = original_img_pil.copy().convert("RGB")
    segments = segmentation_model(img_pil_rgb)

    torso_labels = ["Upper-clothes", "Dress", "Belt", "Face", "Left-arm", "Right-arm"] # Note: Includes "Face"
    
    combined_mask_np = np.zeros((img_pil_rgb.height, img_pil_rgb.width), dtype=np.uint8)
    for s in segments:
        if s['label'] in torso_labels:
            mask_np = np.array(s['mask'].convert('L'))
            combined_mask_np = np.maximum(combined_mask_np, mask_np)

    combined_mask_pil_l = Image.fromarray(combined_mask_np, mode='L')

    # Mask face from the torso mask AFTER combining torso parts
    # This is because "Face" is in torso_labels. If you want to always exclude face,
    # you could remove "Face" from torso_labels and call mask_face unconditionally,
    # or just call mask_face as it is now.
    final_mask_pil_l = mask_face(img_pil_rgb, combined_mask_pil_l)

    img_rgba = img_pil_rgb.copy()
    img_rgba.putalpha(final_mask_pil_l)

    return img_rgba, final_mask_pil_l