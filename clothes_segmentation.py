# ClothesSegmentation.py
from transformers import pipeline
from PIL import Image
import numpy as np

# Initialize model (outside function to load only once)
print("Initializing segmentation_pipeline for clothes_segmentation...")
segmentation_pipeline_clothes = pipeline(model="mattmdjaga/segformer_b2_clothes") # Use a different variable name if it's a separate instance
print("Segmentation_pipeline for clothes_segmentation initialized.")

def segment_clothing(image_pil_rgb, clothing_items=["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]):
    # Ensure image is RGB
    img_to_segment = image_pil_rgb.copy().convert("RGB")

    segmented_output = segmentation_pipeline_clothes(img_to_segment)
    
    # Initialize an empty mask of the correct size
    combined_mask_np = np.zeros((img_to_segment.height, img_to_segment.width), dtype=np.uint8)
    
    found_mask = False
    for segment in segmented_output:
        if segment['label'] in clothing_items:
            # segment['mask'] is a PIL Image, convert to L mode then to numpy array
            mask_np = np.array(segment['mask'].convert('L')) 
            combined_mask_np = np.maximum(combined_mask_np, mask_np)
            found_mask = True

    # If no relevant clothing items were found, combined_mask_np will be all zeros.
    # This is fine, combined_mask_image will be a black mask.
    combined_mask_image_pil_l = Image.fromarray(combined_mask_np, mode='L')
    
    # Create an RGBA image by putting the mask into the alpha channel of the original RGB image
    img_rgba = img_to_segment.copy() # Start with the RGB image (or original image_pil_rgb if preferred)
    img_rgba.putalpha(combined_mask_image_pil_l) # Add the mask as alpha

    return img_rgba # Return the image with alpha channel