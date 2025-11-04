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
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            safety_checker=None,  # Disable for faster generation
        )
        
        # Use DPM solver for faster generation
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Enable attention slicing for memory efficiency
        pipeline.enable_attention_slicing()
        
        # Enable CPU offload for large models on limited VRAM
        if device == "cuda":
            try:
                pipeline.enable_model_cpu_offload()
            except Exception as e:
                logger.warning(f"Could not enable CPU offload: {e}")
        
        logger.info("Pipeline initialized successfully")
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
        
        # Save image
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
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
    """Internal function for image generation"""
    negative_prompt = kwargs.get('negative_prompt', 'blurry, low quality, distorted')
    num_inference_steps = kwargs.get('num_inference_steps', 20)
    guidance_scale = kwargs.get('guidance_scale', 7.5)
    width = kwargs.get('width', 512)
    height = kwargs.get('height', 512)
    seed = kwargs.get('seed', None)
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
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
