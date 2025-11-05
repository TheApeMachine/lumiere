#!/usr/bin/env python3
"""
Audio Analysis Service using librosa
Analyzes audio for intensity, tempo, beats, and other features
"""

import os
import librosa
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'audio-analyzer'
    })

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Analyze audio file for various features"""
    try:
        data = request.json
        audio_path = data.get('audio_path')
        
        if not audio_path:
            return jsonify({'error': 'audio_path is required'}), 400
        
        if not os.path.exists(audio_path):
            return jsonify({'error': f'Audio file not found: {audio_path}'}), 400
        
        logger.info(f"Analyzing audio: {audio_path}")
        
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        logger.info(f"Audio loaded: duration={duration:.2f}s, sample_rate={sr}Hz")
        
        # Tempo and beat detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Onset strength (energy/intensity)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Sample intensity at regular intervals (1 second)
        times = librosa.times_like(onset_env, sr=sr)
        intensity_curve = []
        
        for t in np.arange(0, duration, 1.0):
            idx = np.argmin(np.abs(times - t))
            intensity = onset_env[idx].item() if hasattr(onset_env[idx], 'item') else float(onset_env[idx])
            # Normalize to 0-1 range
            intensity_normalized = min(1.0, intensity / 10.0)
            intensity_curve.append({
                'timestamp': float(t),
                'value': intensity_normalized
            })
        
        # Detect key moments (peaks in onset strength)
        peaks = librosa.util.peak_pick(onset_env, 
                                       pre_max=3, 
                                       post_max=3, 
                                       pre_avg=3, 
                                       post_avg=5, 
                                       delta=0.5, 
                                       wait=10)
        
        peak_times = librosa.frames_to_time(peaks, sr=sr)
        
        # Select strategic key moments throughout the song
        key_moments = select_key_moments(duration, peak_times, intensity_curve)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate (texture)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        result = {
            'duration': float(duration),
            'sample_rate': int(sr),
            'tempo': float(tempo.item()) if hasattr(tempo, 'item') else float(tempo),
            'beat_times': [float(t) for t in beat_times.tolist()[:50]],  # First 50 beats
            'num_beats': len(beat_times),
            'intensity_curve': intensity_curve,
            'key_moments': key_moments,
            'spectral_features': {
                'mean_centroid': float(np.mean(spectral_centroids)),
                'mean_rolloff': float(np.mean(spectral_rolloff)),
                'mean_zcr': float(np.mean(zcr))
            }
        }
        
        tempo_value = tempo.item() if hasattr(tempo, 'item') else float(tempo)
        logger.info(f"Analysis complete: tempo={tempo_value:.1f} BPM, {len(key_moments)} key moments")
        
        return jsonify({
            'success': True,
            'analysis': result
        })
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        return jsonify({'error': str(e)}), 500

def select_key_moments(duration, peak_times, intensity_curve):
    """Select 7 strategic key moments throughout the song"""
    key_moments = []
    
    # Define proportions for key moments
    # 0: Start, 15%, 33%, 50%, 67%, 85%, 100%: End
    proportions = [0.0, 0.15, 0.33, 0.50, 0.67, 0.85, 1.0]
    descriptions = [
        "opening scene, establishing atmosphere",
        "building energy, introducing elements",
        "first climax, dynamic action",
        "bridge, contemplative moment",
        "building to crescendo",
        "peak moment, maximum intensity",
        "resolution, closing scene"
    ]
    
    for i, prop in enumerate(proportions):
        target_time = duration * prop
        
        # Find nearest peak to target time, or use target time if no peaks nearby
        if len(peak_times) > 0:
            distances = np.abs(peak_times - target_time)
            nearest_idx = np.argmin(distances)
            
            # Only use peak if it's within 10 seconds of target
            if distances[nearest_idx] < 10.0:
                timestamp = peak_times[nearest_idx].item() if hasattr(peak_times[nearest_idx], 'item') else float(peak_times[nearest_idx])
            else:
                timestamp = target_time
        else:
            timestamp = target_time
        
        # Get intensity at this timestamp
        intensity_idx = min(int(timestamp), len(intensity_curve) - 1)
        intensity = intensity_curve[intensity_idx]['value'] if intensity_idx < len(intensity_curve) else 0.5
        
        key_moments.append({
            'timestamp': float(timestamp),
            'description': descriptions[i],
            'intensity': float(intensity)
        })
    
    return key_moments

if __name__ == '__main__':
    logger.info("Starting Audio Analysis Service")
    port = int(os.environ.get('AUDIO_SERVICE_PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False)
