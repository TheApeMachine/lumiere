#!/usr/bin/env python3
"""
Video Generation Service using LTX-Video
Supports both CUDA and Metal (MPS) acceleration for consumer hardware
"""

import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import logging
import cv2
import numpy as np

# Try to import LTX-Video (Diffusers integration)
try:
    from diffusers import LTXConditionPipeline
    from diffusers.utils import export_to_video, load_image
    LTX_AVAILABLE = True
except ImportError:
    LTX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LTX-Video not available, falling back to interpolation")

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
    """Initialize LTX-Video pipeline with appropriate device"""
    global pipeline, device
    
    device = get_device()
    logger.info(f"Initializing LTX-Video pipeline on device: {device}")
    
    try:
        if device == "cpu":
            logger.warning("CPU detected - will use frame interpolation instead of AI models")
            return True
                
        # Use LTX-Video model (this will auto-download from HuggingFace)
        model_id = "Lightricks/LTX-Video"  # Use the dev version as you suggested
        
        logger.info(f"Downloading LTX-Video model: {model_id}")
        logger.info("This may take several minutes on first run...")
        
        pipeline = LTXConditionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device in ["cuda", "mps"] else torch.float32,
        )
        
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations
        pipeline.vae.enable_tiling()
        
        # Enable CPU offload for CUDA to save VRAM
        if device == "cuda":
            try:
                pipeline.enable_model_cpu_offload()
            except Exception as e:
                logger.warning(f"Could not enable CPU offload: {e}")
        
        logger.info("LTX-Video pipeline initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LTX-Video pipeline: {e}")
        logger.info("Falling back to interpolation mode")
        return True

def simple_interpolate_frames(start_frame_path, end_frame_path, output_path, num_frames=14, fps=7):
    """Enhanced frame interpolation with smooth transitions"""
    logger.info("Using enhanced interpolation for video generation")
    
    # Load images
    start_img = cv2.imread(start_frame_path)
    end_img = cv2.imread(end_frame_path)
    
    if start_img is None or end_img is None:
        raise ValueError("Could not load start or end frame")
    
    # Ensure same size
    if start_img.shape != end_img.shape:
        end_img = cv2.resize(end_img, (start_img.shape[1], start_img.shape[0]))
    
    height, width = start_img.shape[:2]
    
    # Create video writer with better codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate interpolated frames with smooth easing
    for i in range(num_frames):
        # Use smooth easing instead of linear interpolation
        t = i / (num_frames - 1)
        # Smooth step function for more natural transitions
        alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
        
        # Basic blend
        frame = cv2.addWeighted(start_img, 1 - alpha, end_img, alpha, 0)
        
        # Add subtle zoom effect for more dynamic feel
        if i < num_frames // 2:
            # Slight zoom in during first half
            zoom_factor = 1.0 + (alpha * 0.05)  # Max 5% zoom
        else:
            # Slight zoom out during second half  
            zoom_factor = 1.05 - ((alpha - 0.5) * 0.05)
        
        if zoom_factor != 1.0:
            h_new, w_new = int(height * zoom_factor), int(width * zoom_factor)
            frame_zoomed = cv2.resize(frame, (w_new, h_new))
            
            # Center crop back to original size
            y_start = (h_new - height) // 2
            x_start = (w_new - width) // 2
            frame = frame_zoomed[y_start:y_start+height, x_start:x_start+width]
        
        out.write(frame)
    
    out.release()
    logger.info(f"Video created with {num_frames} interpolated frames at {output_path}")
    return output_path

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': device,
        'pipeline_loaded': pipeline is not None,
        'ltx_available': LTX_AVAILABLE,
        'mode': 'ltx-video' if (pipeline and LTX_AVAILABLE) else 'interpolation'
    })

@app.route('/generate', methods=['POST'])
def generate_video():
    """Generate video from start and end frames"""
    try:
        data = request.json
        start_frame_path = data.get('start_frame')
        end_frame_path = data.get('end_frame')
        output_path = data.get('output_path')
        num_frames = data.get('num_frames', 14)
        fps = data.get('fps', 7)
        
        if not start_frame_path or not end_frame_path:
            return jsonify({'error': 'start_frame and end_frame are required'}), 400
        
        if not output_path:
            return jsonify({'error': 'output_path is required'}), 400
        
        logger.info(f"Generating video from {start_frame_path} to {end_frame_path}")
        
        # Check if files exist
        if not os.path.exists(start_frame_path):
            return jsonify({'error': f'Start frame not found: {start_frame_path}'}), 400
        if not os.path.exists(end_frame_path):
            return jsonify({'error': f'End frame not found: {end_frame_path}'}), 400
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use AI pipeline if available, otherwise use interpolation
        if pipeline and device != "cpu":
            try:
                result_path = generate_with_ai(
                    start_frame_path, end_frame_path, output_path, 
                    num_frames, fps
                )
            except Exception as e:
                logger.warning(f"AI generation failed: {e}, falling back to interpolation")
                result_path = simple_interpolate_frames(
                    start_frame_path, end_frame_path, output_path, 
                    num_frames, fps
                )
        else:
            result_path = simple_interpolate_frames(
                start_frame_path, end_frame_path, output_path, 
                num_frames, fps
            )
        
        logger.info(f"Video saved to: {result_path}")
        return jsonify({
            'success': True,
            'output_path': result_path,
            'num_frames': num_frames,
            'fps': fps
        })
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return jsonify({'error': str(e)}), 500

def generate_with_ai(start_frame_path, end_frame_path, output_path, num_frames, fps):
    """Generate video using LTX-Video model"""
    logger.info("Using LTX-Video model for video generation")
    
    if not LTX_AVAILABLE or pipeline is None:
        raise Exception("LTX-Video pipeline not available")
    
    # Load start frame
    image = load_image(start_frame_path)
    
    # LTX-Video works best with resolutions divisible by 32
    # Keep it reasonable for consumer hardware
    target_width, target_height = 512, 512
    image = image.resize((target_width, target_height))
    
    # LTX-Video works with frames divisible by 8 + 1
    # Adjust num_frames to fit this requirement
    adjusted_frames = ((num_frames - 1) // 8) * 8 + 1
    if adjusted_frames < 9:
        adjusted_frames = 9  # Minimum
    if adjusted_frames > 25:
        adjusted_frames = 25  # Keep it reasonable for speed
    
    # Create a simple prompt based on the transition
    prompt = "smooth cinematic transition, high quality, detailed"
    negative_prompt = "blurry, low quality, distorted, jittery, inconsistent motion"
    
    # Generate video with LTX-Video
    with torch.inference_mode():
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from diffusers.utils import load_video
        
        # Convert image to video format for conditioning
        video = load_video(export_to_video([image], fps=fps))
        condition = LTXVideoCondition(video=video, frame_index=0)
        
        result = pipeline(
            conditions=[condition],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=target_width,
            height=target_height,
            num_frames=adjusted_frames,
            num_inference_steps=10,  # Fast generation
            generator=torch.Generator(device=device).manual_seed(42),
            output_type="pil"
        )
        
        frames = result.frames[0]
    
    # Export to video
    export_to_video(frames, output_path, fps=fps)
    
    return output_path

@app.route('/interpolate', methods=['POST'])
def interpolate():
    """Simple frame interpolation endpoint"""
    try:
        data = request.json
        start_frame = data.get('start_frame')
        end_frame = data.get('end_frame')
        output_path = data.get('output_path')
        num_frames = data.get('num_frames', 14)
        fps = data.get('fps', 7)
        
        if not all([start_frame, end_frame, output_path]):
            return jsonify({'error': 'start_frame, end_frame, and output_path are required'}), 400
        
        result = simple_interpolate_frames(start_frame, end_frame, output_path, num_frames, fps)
        
        return jsonify({
            'success': True,
            'output_path': result
        })
        
    except Exception as e:
        logger.error(f"Error in interpolation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Video Generation Service")
    logger.info(f"Device detection: {get_device()}")
    
    # Initialize pipeline on startup
    if initialize_pipeline():
        port = int(os.environ.get('VIDEO_SERVICE_PORT', 5002))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Failed to initialize. Exiting.")
        exit(1)
