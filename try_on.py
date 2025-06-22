# virtual_try_on/try_on.py

from diffusers import AutoPipelineForInpainting, AutoencoderKL, DPMSolverMultistepScheduler # MODIFIED: Added DPMSolverMultistepScheduler
import torch
from PIL import Image
from .body_segmentation import apply_segmentation

# Determine the device
if torch.cuda.is_available():
    device = "cuda"
    main_pipeline_dtype = torch.float16
    vae_dtype = torch.float16
    pipeline_variant = "fp16"
    print("CUDA is available. Using GPU.")
    DEFAULT_INFERENCE_STEPS = 50
else:
    device = "cpu"
    main_pipeline_dtype = torch.float32
    vae_dtype = torch.float32
    pipeline_variant = None
    print("CUDA not available. Using CPU. This will be slow and require more RAM.")
    DEFAULT_INFERENCE_STEPS = 15

# Load VAE
try:
    print(f"Loading VAE with dtype: {vae_dtype}")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=vae_dtype
    ).to(device)
    print("VAE loaded successfully.")
except Exception as e:
    print(f"Could not load VAE 'madebyollin/sdxl-vae-fp16-fix' with dtype {vae_dtype}: {e}")
    print("Attempting to let the main pipeline load its default VAE.")
    vae = None

# Load Inpainting Pipeline
print(f"Loading Inpainting Pipeline with dtype: {main_pipeline_dtype} and variant: {pipeline_variant}")

# --- MODIFIED: Configure and use DPMSolverMultistepScheduler ---
# First, get the config from the default scheduler of the pipeline if we were to load it normally
temp_pipeline_config = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", use_safetensors=True, torch_dtype=main_pipeline_dtype
).scheduler.config # Get default scheduler config

# Create an instance of DPMSolverMultistepScheduler with that config
# and ensure it uses settings appropriate for SDXL quality
dpm_scheduler = DPMSolverMultistepScheduler.from_config(
    temp_pipeline_config,
    algorithm_type="sde-dpmsolver++", # Good for SDXL
    use_karras_sigmas=True # Often improves quality
)
print(f"Using DPMSolverMultistepScheduler: {dpm_scheduler}")
# --- END OF SCHEDULER MODIFICATION ---

pipeline_kwargs = {
    "torch_dtype": main_pipeline_dtype,
    "use_safetensors": True,
    "scheduler": dpm_scheduler # MODIFIED: Pass the new scheduler instance
}
if pipeline_variant:
    pipeline_kwargs["variant"] = pipeline_variant
if vae is not None:
    pipeline_kwargs["vae"] = vae

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    **pipeline_kwargs
).to(device)
print("Inpainting Pipeline loaded successfully with custom DPM scheduler.")

# The explicit scheduler.to(device) might not be needed with DPM but doesn't hurt
if hasattr(pipeline, 'scheduler') and hasattr(pipeline.scheduler, 'to'):
    try:
        pipeline.scheduler = pipeline.scheduler.to(device)
        if hasattr(pipeline.scheduler, 'alphas_cumprod'): # DPM might not have this exact attribute
            print(f"Scheduler's alphas_cumprod device after explicit move: {pipeline.scheduler.alphas_cumprod.device}")
        else:
            print(f"Scheduler (DPM) moved to {device}. Note: DPM schedulers may not use 'alphas_cumprod' directly in the same way.")
    except Exception as e_scheduler_move:
        print(f"Warning: Could not explicitly move DPM scheduler to device or verify: {e_scheduler_move}")

# --- IP ADAPTER LOADING ---
print("Attempting to load IP Adapter...")
ip_adapter_loaded_successfully = False
try:
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin"
    )
    if hasattr(pipeline, 'image_encoder') and pipeline.image_encoder is not None:
         pipeline.image_encoder.to(device)
    ip_adapter_loaded_successfully = True
    print("IP Adapter loaded successfully.")
except Exception as e:
    print(f"Could not load IP Adapter: {e}")
    print("IP Adapter loading failed.")

def virtual_try_on(img, clothing_ip_image, prompt, negative_prompt, ip_scale=0.7, strength=0.90, guidance_scale=7.5, steps=None):
    if steps is None:
        steps = DEFAULT_INFERENCE_STEPS
    
    print(f"Using {steps} inference steps on {device} with DPM scheduler.")

    if not isinstance(img, Image.Image):
        raise TypeError("Input 'img' must be a PIL Image.")
    if not isinstance(clothing_ip_image, Image.Image):
        raise TypeError("Input 'clothing_ip_image' must be a PIL Image.")
        
    img = img.convert("RGB")
    clothing_ip_image = clothing_ip_image.convert("RGB")

    print("Applying body segmentation for mask...")
    _, mask_img_pil = apply_segmentation(img.copy(), include_face=False)
    mask_img_pil = mask_img_pil.convert("L")

    print("Starting virtual try-on inference...")
    pipeline_call_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": img,
        "mask_image": mask_img_pil,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
    }

    if ip_adapter_loaded_successfully:
        print(f"Using IP Adapter with scale: {ip_scale}")
        pipeline.set_ip_adapter_scale(ip_scale)
        pipeline_call_args["ip_adapter_image"] = clothing_ip_image
    else:
       print("Skipping ip_adapter_image.")

    generated_images = pipeline(**pipeline_call_args).images
    print("Inference complete.")
    return generated_images[0]