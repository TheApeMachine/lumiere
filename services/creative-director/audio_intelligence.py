"""
Audio Intelligence Module - Advanced audio analysis for creative insights.

This module provides comprehensive audio analysis including:
- Genre classification
- Instrument recognition
- Tempo and rhythm analysis
- Mood and energy detection
- Lyric processing and theme extraction
"""

import librosa
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from dataclasses import dataclass
import re
from collections import Counter
import scipy.ndimage
from scipy.signal import find_peaks


class AudioFeatures(BaseModel):
    """Extracted audio features for analysis."""
    tempo: float
    key: str
    mode: str  # major/minor
    time_signature: str
    spectral_centroid: np.ndarray
    mfcc: np.ndarray
    chroma: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    rms_energy: np.ndarray
    onset_times: np.ndarray
    beats: np.ndarray

    class Config:
        arbitrary_types_allowed = True


@dataclass
class MusicalSegment:
    """A segment of music with specific characteristics."""
    start_time: float
    end_time: float
    tempo: float
    energy: float
    instruments: List[str]
    mood: str
    description: str


class AudioIntelligence:
    """Advanced audio analysis for creative music video generation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Genre classification mapping based on audio features
        self.genre_classifiers = {
            'electronic': {'tempo_range': (120, 140), 'spectral_features': 'high_frequency'},
            'rock': {'tempo_range': (110, 130), 'spectral_features': 'mid_frequency'},
            'hip_hop': {'tempo_range': (70, 100), 'spectral_features': 'bass_heavy'},
            'pop': {'tempo_range': (100, 130), 'spectral_features': 'balanced'},
            'classical': {'tempo_range': (60, 120), 'spectral_features': 'harmonic_rich'},
            'jazz': {'tempo_range': (80, 140), 'spectral_features': 'complex_harmony'},
            'ambient': {'tempo_range': (60, 90), 'spectral_features': 'atmospheric'}
        }

        # Instrument frequency signatures (simplified)
        self.instrument_signatures = {
            'drums': {'freq_range': (20, 200), 'onset_density': 'high'},
            'bass': {'freq_range': (20, 250), 'onset_density': 'medium'},
            'guitar': {'freq_range': (80, 1200), 'onset_density': 'medium'},
            'piano': {'freq_range': (27, 4200), 'onset_density': 'variable'},
            'synthesizer': {'freq_range': (50, 8000), 'onset_density': 'variable'},
            'vocals': {'freq_range': (80, 1100), 'onset_density': 'lyrical'},
            'strings': {'freq_range': (196, 2093), 'onset_density': 'sustained'},
            'brass': {'freq_range': (146, 1175), 'onset_density': 'punctuated'},
            'woodwinds': {'freq_range': (261, 2093), 'onset_density': 'melodic'}
        }

    def analyze_audio_file(self, file_path: str, sr: int = 22050) -> Dict[str, Any]:
        """
        Comprehensive analysis of an audio file.

        Args:
            file_path: Path to the audio file
            sr: Sample rate for analysis

        Returns:
            Complete analysis results
        """
        try:
            self.logger.info(f"Starting audio analysis for: {file_path}")

            # Load audio file
            y, sr = librosa.load(file_path, sr=sr)
            duration = librosa.get_duration(y=y, sr=sr)

            # Extract features
            features = self._extract_audio_features(y, sr)

            # Analyze musical elements
            genre = self._classify_genre(features)
            tempo_analysis = self._analyze_tempo(features, y, sr)
            instruments = self._detect_instruments(features, y, sr)
            mood_energy = self._analyze_mood_and_energy(features, y, sr)
            structure = self._analyze_song_structure(y, sr)
            key_moments = self._identify_key_moments(y, sr, features)

            analysis_result = {
                'file_info': {
                    'duration': duration,
                    'sample_rate': sr,
                    'file_path': file_path
                },
                'musical_analysis': {
                    'genre': genre,
                    'tempo': tempo_analysis,
                    'key': features.key,
                    'mode': features.mode,
                    'time_signature': features.time_signature
                },
                'instruments': instruments,
                'mood_and_energy': mood_energy,
                'song_structure': structure,
                'key_moments': key_moments
            }

            self.logger.info("Audio analysis completed successfully")
            return analysis_result

        except Exception as e:
            self.logger.error(f"Error analyzing audio file: {e}")
            raise

    def _extract_audio_features(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Extract comprehensive audio features."""

        # Tempo and beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Harmonic analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # Key detection (simplified)
        key_profiles = np.array([
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # C major
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],  # C minor
        ])
        chroma_mean = np.mean(chroma, axis=1)
        key_correlations = np.dot(key_profiles, chroma_mean)
        key_idx = np.argmax(key_correlations)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        modes = ['major', 'minor']

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        rms_energy = librosa.feature.rms(y=y)

        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        return AudioFeatures(
            tempo=tempo,
            key=keys[key_idx % 12],
            mode=modes[key_idx // 12],
            time_signature="4/4",  # Simplified
            spectral_centroid=spectral_centroid,
            mfcc=mfcc,
            chroma=chroma,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zero_crossing_rate,
            rms_energy=rms_energy,
            onset_times=onset_times,
            beats=beat_times,
        )

    def _classify_genre(self, features: AudioFeatures) -> Dict[str, Any]:
        """Classify the genre based on audio features."""
        tempo = features.tempo
        spectral_centroid_mean = np.mean(features.spectral_centroid)
        genre_scores = {}

        for genre, characteristics in self.genre_classifiers.items():
            score = 0
            tempo_range = characteristics['tempo_range']
            if tempo_range[0] <= tempo <= tempo_range[1]:
                score += 0.4
            else:
                distance = min(abs(tempo - tempo_range[0]), abs(tempo - tempo_range[1]))
                score += max(0, 0.4 - distance / 50)
            if characteristics['spectral_features'] == 'high_frequency' and spectral_centroid_mean > 3000:
                score += 0.3
            elif characteristics['spectral_features'] == 'mid_frequency' and 1000 < spectral_centroid_mean < 3000:
                score += 0.3
            elif characteristics['spectral_features'] == 'bass_heavy' and spectral_centroid_mean < 1500:
                score += 0.3
            genre_scores[genre] = score

        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        return {
            'primary_genre': sorted_genres[0][0],
            'confidence': sorted_genres[0][1],
            'secondary_genres': [g[0] for g in sorted_genres[1:3]],
            'all_scores': genre_scores
        }

    def _analyze_tempo(self, features: AudioFeatures, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze tempo and rhythm characteristics."""
        tempo = features.tempo
        if tempo < 60: tempo_class = "very_slow"
        elif tempo < 90: tempo_class = "slow"
        elif tempo < 120: tempo_class = "moderate"
        elif tempo < 140: tempo_class = "fast"
        else: tempo_class = "very_fast"

        tempo_track = librosa.feature.rhythm.tempo(y=y, sr=sr, aggregate=None)
        tempo_variance = np.var(tempo_track) if len(tempo_track) > 1 else 0
        stability = "stable" if tempo_variance < 100 else "variable"

        return {
            'bpm': float(tempo),
            'classification': tempo_class,
            'stability': stability,
            'variance': float(tempo_variance)
        }

    def _detect_instruments(self, features: AudioFeatures, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect instruments present in the audio."""
        detected_instruments = []
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)

        for instrument, signature in self.instrument_signatures.items():
            confidence = 0
            freq_range = signature['freq_range']
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            if np.any(freq_mask):
                energy_in_range = np.mean(magnitude[freq_mask, :])
                total_energy = np.mean(magnitude)
                if total_energy > 0:
                    freq_ratio = energy_in_range / total_energy
                    confidence += freq_ratio * 0.6

            onset_density = len(features.onset_times) / (len(y) / sr)
            if signature['onset_density'] == 'high' and onset_density > 2: confidence += 0.4
            elif signature['onset_density'] == 'medium' and 0.5 < onset_density < 2: confidence += 0.4
            elif signature['onset_density'] == 'low' and onset_density < 0.5: confidence += 0.4

            if confidence > 0.3:
                detected_instruments.append({
                    'instrument': instrument,
                    'confidence': float(confidence),
                    'prominence': 'high' if confidence > 0.7 else 'medium'
                })

        detected_instruments.sort(key=lambda x: x['confidence'], reverse=True)
        return detected_instruments

    def _analyze_mood_and_energy(self, features: AudioFeatures, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze mood and energy characteristics."""
        rms_mean = np.mean(features.rms_energy)
        rms_var = np.var(features.rms_energy)
        spectral_centroid_mean = np.mean(features.spectral_centroid)

        if rms_mean > 0.1: energy_level = "high"
        elif rms_mean > 0.05: energy_level = "medium"
        else: energy_level = "low"

        mood_indicators = {'happy': 0, 'sad': 0, 'energetic': 0, 'calm': 0, 'aggressive': 0, 'mysterious': 0}
        if features.mode == 'major':
            mood_indicators['happy'] += 0.3
            mood_indicators['energetic'] += 0.2
        else:
            mood_indicators['sad'] += 0.3
            mood_indicators['mysterious'] += 0.2

        if features.tempo > 120:
            mood_indicators['energetic'] += 0.3
            mood_indicators['happy'] += 0.2
        elif features.tempo < 80:
            mood_indicators['calm'] += 0.3
            mood_indicators['sad'] += 0.2

        if energy_level == 'high':
            mood_indicators['energetic'] += 0.3
            mood_indicators['aggressive'] += 0.2
        elif energy_level == 'low':
            mood_indicators['calm'] += 0.3

        dominant_mood = max(mood_indicators.items(), key=lambda x: x[1])
        return {
            'energy_level': energy_level,
            'energy_variance': float(rms_var),
            'dominant_mood': dominant_mood[0],
            'mood_confidence': float(dominant_mood[1]),
            'all_moods': mood_indicators,
            'spectral_brightness': float(spectral_centroid_mean)
        }

    def _analyze_song_structure(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze the structural elements of the song."""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        boundaries = librosa.segment.agglomerative(chroma, k=8)
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)
        segments = []
        for i in range(len(boundary_times) - 1):
            start_time, end_time = boundary_times[i], boundary_times[i+1]
            duration = end_time - start_time
            if i == 0: segment_type = "intro"
            elif i == len(boundary_times) - 2: segment_type = "outro"
            elif duration > 30: segment_type = "verse" if i % 2 == 1 else "chorus"
            else: segment_type = "bridge"
            segments.append({'type': segment_type, 'start_time': float(start_time), 'end_time': float(end_time), 'duration': float(duration)})

        return {
            'segments': segments,
            'total_segments': len(segments),
            'structure_complexity': 'simple' if len(segments) < 6 else 'complex'
        }

    def _identify_key_moments(self, y: np.ndarray, sr: int, features: AudioFeatures) -> List[Dict[str, Any]]:
        """Identify key musical moments like drops, transitions, and quiet parts."""
        all_moments = []
        if features.rms_energy is not None and len(features.rms_energy) > 0:
            rms = features.rms_energy[0]
            rms_smooth = scipy.ndimage.median_filter(rms, size=5)
            peaks, _ = find_peaks(rms_smooth, height=np.mean(rms_smooth) + np.std(rms_smooth), distance=5)
            peak_times = librosa.frames_to_time(peaks, sr=sr)
            for peak_time in peak_times:
                all_moments.append({'timestamp': float(peak_time), 'type': 'energy_peak', 'intensity': float(np.interp(rms_smooth[int(peak_time * sr / 512)], [min(rms_smooth), max(rms_smooth)], [0.5, 1.0]))})

        if features.onset_times is not None:
            for onset_time in features.onset_times[::4]:
                all_moments.append({'timestamp': float(onset_time), 'type': 'onset', 'intensity': 0.6})

        if features.beats is not None:
            for beat in features.beats[::8]:
                all_moments.append({'timestamp': float(beat), 'type': 'beat', 'intensity': 0.3})

        all_moments.sort(key=lambda x: x['timestamp'])
        filtered_moments = []
        last_time = -2.0
        for moment in all_moments:
            if moment['timestamp'] > last_time + 2.0:
                filtered_moments.append(moment)
                last_time = moment['timestamp']

        return filtered_moments[:20]

class LyricsIntelligence:
    """Advanced lyrics analysis for thematic and narrative insights."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.theme_keywords = {
            'love': ['love', 'heart', 'kiss', 'romance'], 'freedom': ['free', 'fly', 'escape', 'break'],
            'struggle': ['fight', 'battle', 'war', 'pain'], 'celebration': ['party', 'dance', 'celebrate', 'joy'],
            'journey': ['road', 'travel', 'journey', 'path'], 'dreams': ['dream', 'hope', 'wish', 'believe'],
            'rebellion': ['rebel', 'revolution', 'change', 'system'], 'nostalgia': ['remember', 'past', 'yesterday', 'memories'],
            'spirituality': ['god', 'heaven', 'soul', 'spirit'], 'nature': ['sun', 'moon', 'stars', 'ocean']
        }

    def analyze_lyrics(self, lyrics: str) -> Dict[str, Any]:
        """Comprehensive lyrics analysis."""
        if not lyrics: return {'error': 'No lyrics provided'}
        clean_lyrics = self._clean_lyrics(lyrics)
        themes = self._extract_themes(clean_lyrics)
        return { 'themes': themes, 'word_count': len(clean_lyrics.split()) }

    def _clean_lyrics(self, lyrics: str) -> str:
        """Clean and normalize lyrics text."""
        lyrics = re.sub(r'\s+', ' ', lyrics.strip())
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        return lyrics.lower()

    def _extract_themes(self, lyrics: str) -> Dict[str, Any]:
        """Extract thematic content from lyrics."""
        theme_scores = {}
        for theme, keywords in self.theme_keywords.items():
            score = sum(lyrics.count(keyword) for keyword in keywords)
            if score > 0:
                theme_scores[theme] = {'score': score}

        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return {
            'primary_themes': [theme[0] for theme in sorted_themes[:3]],
            'all_themes': dict(sorted_themes)
        }