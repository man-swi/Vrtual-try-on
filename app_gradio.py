# app_gradio.py
import gradio as gr
from PIL import Image
import torch # For device check
import sys
import os
import time

# --- Ensure project path is in sys.path ---
# This assumes app_gradio.py is in the project root '/content/virtual-try-on-outfit-change/'
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root added to sys.path: {project_root}")

# --- Import your core try-on logic ---
# This import will trigger model loading from try_on.py if they haven't been loaded yet.
# It's crucial that try_on.py is structured to load models into global variables
# so they are loaded only ONCE when this script starts.
print("Importing virtual_try_on.try_on - This will load models (may take time)...")
start_model_load_time = time.time()
from virtual_try_on.try_on import virtual_try_on, device as try_on_device, DEFAULT_INFERENCE_STEPS
# from virtual_try_on.body_segmentation import apply_segmentation # Not directly called by Gradio UI, but used by virtual_try_on
# from virtual_try_on.clothes_segmentation import segment_clothing # Not directly called by Gradio UI
end_model_load_time = time.time()
print(f"Models loaded/checked by try_on.py import in {end_model_load_time - start_model_load_time:.2f} seconds.")
print(f"Try-on will run on device: {try_on_device}")

# --- Define the image processing function for Gradio ---
def process_images_for_try_on(person_pil_image, clothes_pil_image,
                              prompt_text, neg_prompt_text,
                              ip_scale_val, strength_val, steps_val,
                              progress=gr.Progress(track_tqdm=True)):
    """
    Processes the uploaded images and parameters to perform virtual try-on.
    progress object can be used with tqdm if your pipeline supports it.
    """
    if person_pil_image is None or clothes_pil_image is None:
        raise gr.Error("Please upload both person and clothes images.")

    print("\n--- Gradio: New Try-On Request ---")
    print(f"Person image type: {type(person_pil_image)}, Clothes image type: {type(clothes_pil_image)}")
    print(f"Prompt: '{prompt_text}'")
    print(f"IP Scale: {ip_scale_val}, Strength: {strength_val}, Steps: {steps_val}")

    # Gradio's gr.Image(type="pil") should provide PIL images directly.
    # Resize inputs (ensure these are the same sizes your Colab notebook used for good results)
    target_size = (512, 512)
    person_pil_image = person_pil_image.resize(target_size).convert("RGB")
    clothes_pil_image = clothes_pil_image.resize(target_size).convert("RGB")
    print(f"Images resized to {target_size}")

    # Call your core virtual_try_on function
    # The `virtual_try_on` function from your try_on.py should handle all model inference.
    try:
        start_inference_time = time.time()
        result_pil_image = virtual_try_on(
            img=person_pil_image,
            clothing_ip_image=clothes_pil_image,
            prompt=prompt_text,
            negative_prompt=neg_prompt_text,
            ip_scale=float(ip_scale_val),
            strength=float(strength_val),
            steps=int(steps_val)
        )
        end_inference_time = time.time()
        print(f"Gradio: Inference completed in {end_inference_time - start_inference_time:.2f} seconds.")
        return result_pil_image
    except Exception as e:
        print(f"Error during Gradio try-on process: {e}")
        import traceback
        traceback.print_exc()
        # Raise a Gradio error to display it in the UI
        raise gr.Error(f"An error occurred: {str(e)}")


# --- Define the Gradio Interface ---
print("Defining Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ Virtual Try-On AI 👕")
    gr.Markdown("Upload a photo of a person and an image of a clothing item. The AI will generate an image of the person wearing the clothes!")

    with gr.Row():
        with gr.Column(scale=1):
            person_input = gr.Image(type="pil", label="🧍 Person Image", sources=["upload", "webcam", "clipboard"])
            clothes_input = gr.Image(type="pil", label="👚 Clothes Image (garment on white background preferred)", sources=["upload", "webcam", "clipboard"])
            run_button = gr.Button("✨ Generate Try-On ✨", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="🌟 Try-On Result")
            # You could add a gr.Label here to show processing time or status

    with gr.Accordion("⚙️ Advanced Settings", open=False):
        prompt_input = gr.Textbox(
            label="📝 Prompt",
            value="A realistic photo of a person wearing the provided clothes, high quality, good fit, photorealistic",
            lines=2
        )
        neg_prompt_input = gr.Textbox(
            label="🚫 Negative Prompt",
            value="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, bad clothing, distorted clothing, poorly fitting clothing, naked, text, watermark, signature, blur, low resolution, jpeg artifacts, compression artifacts, poorly drawn, cartoon, anime, sketch",
            lines=2
        )
        with gr.Row():
            ip_scale_slider = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.7, label="🧥 IP Adapter Scale (Clothing Adherence)")
            strength_slider = gr.Slider(minimum=0.5, maximum=1.0, step=0.05, value=0.85, label="🎨 Inpainting Strength (Amount of Change)")
        steps_slider = gr.Slider(minimum=20, maximum=100, step=1, value=DEFAULT_INFERENCE_STEPS, label="🔄 Inference Steps (Quality vs. Speed)")

    # Connect the button to the function
    run_button.click(
        fn=process_images_for_try_on,
        inputs=[
            person_input, clothes_input,
            prompt_input, neg_prompt_input,
            ip_scale_slider, strength_slider, steps_slider
        ],
        outputs=output_image
    )

    gr.Markdown("---")
    gr.Markdown("Built with Diffusers, IP-Adapter, and Gradio. Project for Final Year Student.")

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("Models should have been loaded during import of try_on.py.")
    print("Launching Gradio demo...")
    # share=True creates a public link (valid for 72 hours from Gradio's servers)
    # debug=True provides more detailed logs and error messages in the console.
    # In Colab, you might need to set inline=False if it doesn't display correctly,
    # or simply rely on the public link.
    demo.launch(share=True, debug=True)