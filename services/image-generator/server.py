#!/usr/bin/env python3
"""
Image Generation Service using Stable Diffusion
Supports both CUDA and Metal (MPS) acceleration for consumer hardware
"""

import os
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global pipeline variable
pipeline = None
device = None

def get_device():
    """Detect and return the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def initialize_pipeline():
    """Initialize Stable Diffusion pipeline with appropriate device"""
    global pipeline, device
    
    device = get_device()
    logger.info(f"Initializing Stable Diffusion on device: {device}")
    
    # Use smaller model for consumer hardware
    # Stable Diffusion 1.5 is a good balance of quality and performance
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        # For MPS (Apple Silicon), use float32 to avoid potential issues
        dtype = torch.float32 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32)
        logger.info(f"Using dtype: {dtype}")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # Disable for faster generation
            requires_safety_checker=False,
        )
        
        # Use DPM solver for faster generation
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Enable attention slicing for memory efficiency
        pipeline.enable_attention_slicing()
        
        # For MPS, don't use CPU offload as it can cause issues
        if device == "cuda":
            try:
                pipeline.enable_model_cpu_offload()
            except Exception as e:
                logger.warning(f"Could not enable CPU offload: {e}")
        
        # Test the pipeline with a simple generation
        logger.info("Testing pipeline with simple generation...")
        with torch.inference_mode():
            test_result = pipeline(
                "a simple red apple",
                num_inference_steps=10,
                width=256,
                height=256,
                generator=torch.Generator(device=device).manual_seed(42)
            )
        
        test_image = test_result.images[0]
        test_array = np.array(test_image)
        logger.info(f"Test image stats: min={test_array.min()}, max={test_array.max()}, mean={test_array.mean():.2f}")
        
        if test_array.max() == 0:
            logger.error("Test generation failed - pipeline producing black images")
            return False
        
        logger.info("Pipeline initialized and tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': device,
        'pipeline_loaded': pipeline is not None
    })

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image from prompt"""
    try:
        data = request.json
        prompt = data.get('prompt')
        output_path = data.get('output_path')
        negative_prompt = data.get('negative_prompt', 'blurry, low quality, distorted')
        num_inference_steps = data.get('num_inference_steps', 20)  # Lower for speed
        guidance_scale = data.get('guidance_scale', 7.5)
        width = data.get('width', 512)
        height = data.get('height', 512)
        seed = data.get('seed', None)
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503
        
        logger.info(f"Generating image with prompt: {prompt}")
        
        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Use internal generation method
        if output_path:
            generate_image_internal(
                prompt=prompt,
                output_path=output_path,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed
            )
            logger.info(f"Image saved to: {output_path}")
            return jsonify({
                'success': True,
                'output_path': output_path,
                'prompt': prompt
            })
        else:
            return jsonify({'error': 'output_path is required'}), 400
            
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-batch', methods=['POST'])
def generate_batch():
    """Generate multiple images from prompts"""
    try:
        data = request.json
        prompts = data.get('prompts', [])
        output_dir = data.get('output_dir')
        
        if not prompts:
            return jsonify({'error': 'Prompts array is required'}), 400
        
        if not output_dir:
            return jsonify({'error': 'output_dir is required'}), 400
        
        results = []
        for i, prompt_data in enumerate(prompts):
            try:
                prompt = prompt_data.get('prompt')
                output_filename = prompt_data.get('filename', f'image_{i}.png')
                output_path = os.path.join(output_dir, output_filename)
                
                # Generate single image
                response = generate_image_internal(
                    prompt=prompt,
                    output_path=output_path,
                    **prompt_data
                )
                results.append({
                    'index': i,
                    'success': True,
                    'output_path': output_path
                })
            except Exception as e:
                logger.error(f"Error generating image {i}: {e}")
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(prompts),
            'successful': sum(1 for r in results if r['success'])
        })
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return jsonify({'error': str(e)}), 500

def generate_image_internal(prompt, output_path, **kwargs):
    """Internal function for image generation with enhanced debugging and retry logic"""
    negative_prompt = kwargs.get('negative_prompt', 'blurry, low quality, distorted')
    num_inference_steps = kwargs.get('num_inference_steps', 20)
    guidance_scale = kwargs.get('guidance_scale', 7.5)
    width = kwargs.get('width', 512)
    height = kwargs.get('height', 512)
    seed = kwargs.get('seed', None)
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate image
    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
    
    image = result.images[0]
    
    # Debug: Check image properties
    img_array = np.array(image)
    logger.info(f"Generated image stats: shape={img_array.shape}, min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")
    
    # Check for completely black images
    if img_array.max() == 0:
        logger.error("Generated image is completely black! This indicates a model issue.")
        # Try to regenerate with different parameters
        logger.info("Attempting regeneration with modified parameters...")
        with torch.inference_mode():
            result = pipeline(
                prompt=f"high quality, detailed, {prompt}",
                negative_prompt=f"{negative_prompt}, black image, dark",
                num_inference_steps=25,  # More steps
                guidance_scale=8.0,  # Higher guidance
                width=width,
                height=height,
                generator=torch.Generator(device=device).manual_seed(42) if generator is None else generator
            )
        image = result.images[0]
        img_array = np.array(image)
        logger.info(f"Retry image stats: shape={img_array.shape}, min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")
    
    # Ensure image has valid pixel values
    if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
        logger.warning("Generated image contains invalid values, clipping to valid range")
        img_array = np.nan_to_num(img_array, nan=128.0, posinf=255.0, neginf=0.0)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        from PIL import Image as PILImage
        image = PILImage.fromarray(img_array)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    
    return output_path

if __name__ == '__main__':
    logger.info("Starting Image Generation Service")
    logger.info(f"Device detection: {get_device()}")
    
    # Initialize pipeline on startup
    if initialize_pipeline():
        port = int(os.environ.get('IMAGE_SERVICE_PORT', 5001))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Failed to initialize pipeline. Exiting.")
        exit(1)
