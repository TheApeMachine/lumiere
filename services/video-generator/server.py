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
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Try to import LTX-Video (Diffusers integration)
try:
    from diffusers import LTXConditionPipeline
    from diffusers.utils import load_image, load_video, export_to_video
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
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
            logger.error("CPU detected - LTX-Video requires GPU; strict mode: disabling /generate")
            return True  # start server; /generate will refuse

        # Use LTX-Video condition pipeline (matches HF example/API)
        model_id = "Lightricks/LTX-Video"

        logger.info(f"Downloading LTX-Video model: {model_id}")
        logger.info("This may take several minutes on first run...")

        dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
        pipeline_local = LTXConditionPipeline.from_pretrained(model_id, torch_dtype=dtype)

        pipeline = pipeline_local.to(device)

        # # Enable memory optimizations
        # pipeline.vae.enable_tiling()

        # # Enable CPU offload for CUDA to save VRAM
        # if device == "cuda":
        #     try:
        #         pipeline.enable_model_cpu_offload()
        #     except Exception as e:
        #         logger.warning(f"Could not enable CPU offload: {e}")

        logger.info("LTX-Video pipeline initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize LTX-Video pipeline: {e}")
        logger.error("Strict mode: disabling /generate endpoint")
        return True  # start server; /generate will refuse

class VideoPromptsRequest(BaseModel):
    project_id: str
    seed_prompts: List[Dict[str, Any]]
    beat_map: Optional[List[Dict[str, Any]]] = None
    audio_file: str
    audio_summary: Dict[str, Any]

class GenerateRequest(BaseModel):
    start_frame: str
    end_frame: str
    output_path: str
    prompt: Optional[str] = None
    num_frames: int = 16
    fps: int = 8
    use_dynamic_shifting: bool = True
    mu: float = 0.9
    seed: Optional[int] = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint (returns 200 only when generator is ready)."""
    ready = bool(LTX_AVAILABLE and pipeline is not None and device != "cpu")
    mode = 'ltx-video' if (pipeline and LTX_AVAILABLE) else None
    payload = {
        'status': 'ready' if ready else 'not_ready',
        'device': device,
        'pipeline_loaded': pipeline is not None,
        'ltx_available': LTX_AVAILABLE,
        'mode': mode
    }
    return (jsonify(payload), 200) if ready else (jsonify(payload), 503)

@app.route('/ready', methods=['GET'])
def ready():
    """Explicit readiness endpoint equivalent to health in strict mode."""
    return health()

@app.route('/generate', methods=['POST'])
def generate_video():
    """Generate a video from a start and end frame."""
    try:
        req = GenerateRequest(**(request.get_json() or {}))

        start_image = Image.open(req.start_frame).convert("RGB")
        end_image = Image.open(req.end_frame).convert("RGB")

        # LTX requires a video condition; create a tiny conditioning clip from the seed images
        # and anchor conditions at first and last frames
        conditioning_path = export_to_video([start_image, end_image], fps=req.fps)
        conditioning_video = load_video(conditioning_path)

        # Adjust frame count to be compatible with LTX (multiple of 8 plus 1)
        adjusted_frames = ((int(req.num_frames) - 1) // 8) * 8 + 1
        if adjusted_frames < 9:
            adjusted_frames = 9
        if adjusted_frames > 33:
            adjusted_frames = 33

        conditions = [
            LTXVideoCondition(video=conditioning_video, frame_index=0),
            LTXVideoCondition(video=conditioning_video, frame_index=min(len(conditioning_video) - 1, 1)),
        ]

        # Prepare generator if seed is provided
        generator = None
        if req.seed is not None:
            generator = torch.manual_seed(req.seed)

        logger.info(f"Generating video with prompt: {req.prompt}, num_frames: {adjusted_frames}, fps: {req.fps}")

        with torch.inference_mode():
            result = pipeline(
                conditions=conditions,
                prompt=req.prompt or "cinematic video",
                num_frames=adjusted_frames,
                num_inference_steps=10,
                output_type="pil",
                generator=generator,
                use_dynamic_shifting=req.use_dynamic_shifting,
                mu=req.mu,
            )

        frames = getattr(result, "frames", None)
        if not frames:
            raise RuntimeError("Pipeline returned no frames")

        export_to_video(frames[0], req.output_path, fps=req.fps)

        logger.info(f"Video saved to: {req.output_path}")
        return jsonify({
            'success': True,
            'output_path': req.output_path,
            'num_frames': req.num_frames,
            'fps': req.fps
        })

    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/video_prompts', methods=['POST'])
def create_video_prompts():
    """Create video prompts from seed prompts and a beat map."""
    try:
        req = VideoPromptsRequest(**(request.get_json() or {}))

        prompts = []
        for i in range(len(req.seed_prompts) - 1):
            start_prompt = req.seed_prompts[i]
            end_prompt = req.seed_prompts[i+1]

            # Default duration if beat_map is not available
            duration = 10.0
            if req.beat_map and len(req.beat_map) > i + 1:
                start_beat = req.beat_map[i]
                end_beat = req.beat_map[i+1]
                duration = end_beat.get('time', 0) - start_beat.get('time', 0)

            prompts.append({
                "start_prompt": start_prompt,
                "end_prompt": end_prompt,
                "duration": duration,
            })

        resp = {"prompts": prompts}
        return jsonify(resp)

    except Exception as e:
        logger.error(f"/v1/video_prompts error: {e}")
        return jsonify({"error": str(e)}), 500

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
