#!/usr/bin/env python3
"""
Video Generation Service using Stable Video Diffusion
Supports both CUDA and Metal (MPS) acceleration for consumer hardware
"""

import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import logging
import cv2
import numpy as np

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
    """Initialize Stable Video Diffusion pipeline with appropriate device"""
    global pipeline, device
    
    device = get_device()
    logger.info(f"Initializing Stable Video Diffusion on device: {device}")
    
    # Use Stable Video Diffusion model
    # This is a smaller model suitable for consumer hardware
    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    
    try:
        # Note: SVD requires significant VRAM, fallback to simple interpolation on CPU/limited VRAM
        if device == "cpu":
            logger.warning("CPU detected - will use simple frame interpolation instead of full SVD")
            return True
        
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        )
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Enable attention slicing for memory efficiency
        pipeline.enable_attention_slicing()
        
        logger.info("Pipeline initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize full pipeline: {e}")
        logger.info("Falling back to simple interpolation mode")
        return True  # Continue anyway with interpolation fallback

def simple_interpolate_frames(start_frame_path, end_frame_path, output_path, num_frames=14, fps=7):
    """Simple frame interpolation when AI model is not available"""
    logger.info("Using simple linear interpolation for video generation")
    
    # Load images
    start_img = cv2.imread(start_frame_path)
    end_img = cv2.imread(end_frame_path)
    
    if start_img is None or end_img is None:
        raise ValueError("Could not load start or end frame")
    
    # Ensure same size
    if start_img.shape != end_img.shape:
        end_img = cv2.resize(end_img, (start_img.shape[1], start_img.shape[0]))
    
    height, width = start_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate interpolated frames
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = cv2.addWeighted(start_img, 1 - alpha, end_img, alpha, 0)
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
        'mode': 'ai' if pipeline else 'interpolation'
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
    """Generate video using AI model"""
    logger.info("Using AI model for video generation")
    
    # Load start frame
    image = load_image(start_frame_path)
    image = image.resize((512, 512))
    
    # Generate frames
    with torch.inference_mode():
        frames = pipeline(
            image,
            num_frames=num_frames,
            decode_chunk_size=8
        ).frames[0]
    
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
