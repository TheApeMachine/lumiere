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
from dataclasses import dataclass
import re
from collections import Counter

# For more advanced analysis, you might want to add:
# import essentia.standard as es
# import tensorflow as tf
# from transformers import pipeline


@dataclass
class AudioFeatures:
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
                'key_moments': key_moments,
                'creative_insights': self._generate_creative_insights(
                    genre, tempo_analysis, instruments, mood_energy, structure
                )
            }
            
            self.logger.info("Audio analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio file: {e}")
            raise
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Extract comprehensive audio features."""
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Harmonic analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Key detection (simplified)
        key_profiles = np.array([
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # C major
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],  # C minor
            # Add more key profiles...
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
            time_signature="4/4",  # Simplified - would need more complex analysis
            spectral_centroid=spectral_centroid,
            mfcc=mfcc,
            chroma=chroma,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zero_crossing_rate,
            rms_energy=rms_energy,
            onset_times=onset_times
        )
    
    def _classify_genre(self, features: AudioFeatures) -> Dict[str, Any]:
        """Classify the genre based on audio features."""
        
        tempo = features.tempo
        spectral_centroid_mean = np.mean(features.spectral_centroid)
        
        genre_scores = {}
        
        for genre, characteristics in self.genre_classifiers.items():
            score = 0
            
            # Tempo matching
            tempo_range = characteristics['tempo_range']
            if tempo_range[0] <= tempo <= tempo_range[1]:
                score += 0.4
            else:
                # Penalty for being outside range
                distance = min(abs(tempo - tempo_range[0]), abs(tempo - tempo_range[1]))
                score += max(0, 0.4 - distance / 50)
            
            # Spectral characteristics (simplified)
            if characteristics['spectral_features'] == 'high_frequency' and spectral_centroid_mean > 3000:
                score += 0.3
            elif characteristics['spectral_features'] == 'mid_frequency' and 1000 < spectral_centroid_mean < 3000:
                score += 0.3
            elif characteristics['spectral_features'] == 'bass_heavy' and spectral_centroid_mean < 1500:
                score += 0.3
            
            # Additional features could include rhythm patterns, harmonic complexity, etc.
            
            genre_scores[genre] = score
        
        # Get top genres
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
        
        # Classify tempo
        if tempo < 60:
            tempo_class = "very_slow"
        elif tempo < 90:
            tempo_class = "slow"
        elif tempo < 120:
            tempo_class = "moderate"
        elif tempo < 140:
            tempo_class = "fast"
        else:
            tempo_class = "very_fast"
        
        # Analyze tempo stability
        tempo_track = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
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
        
        # Analyze frequency content
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr)
        
        for instrument, signature in self.instrument_signatures.items():
            confidence = 0
            
            # Check frequency range presence
            freq_range = signature['freq_range']
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            
            if np.any(freq_mask):
                # Calculate energy in frequency range
                energy_in_range = np.mean(magnitude[freq_mask, :])
                total_energy = np.mean(magnitude)
                
                if total_energy > 0:
                    freq_ratio = energy_in_range / total_energy
                    confidence += freq_ratio * 0.6
            
            # Check onset patterns
            onset_density = len(features.onset_times) / (len(y) / sr)  # onsets per second
            
            if signature['onset_density'] == 'high' and onset_density > 2:
                confidence += 0.4
            elif signature['onset_density'] == 'medium' and 0.5 < onset_density < 2:
                confidence += 0.4
            elif signature['onset_density'] == 'low' and onset_density < 0.5:
                confidence += 0.4
            
            # Only include instruments with reasonable confidence
            if confidence > 0.3:
                detected_instruments.append({
                    'instrument': instrument,
                    'confidence': float(confidence),
                    'prominence': 'high' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'low'
                })
        
        # Sort by confidence
        detected_instruments.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_instruments
    
    def _analyze_mood_and_energy(self, features: AudioFeatures, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze mood and energy characteristics."""
        
        # Energy analysis
        rms_mean = np.mean(features.rms_energy)
        rms_var = np.var(features.rms_energy)
        
        # Spectral characteristics for mood
        spectral_centroid_mean = np.mean(features.spectral_centroid)
        
        # Determine energy level
        if rms_mean > 0.1:
            energy_level = "high"
        elif rms_mean > 0.05:
            energy_level = "medium"
        else:
            energy_level = "low"
        
        # Determine mood based on key, tempo, and spectral features
        mood_indicators = {
            'happy': 0,
            'sad': 0,
            'energetic': 0,
            'calm': 0,
            'aggressive': 0,
            'mysterious': 0
        }
        
        # Key-based mood (simplified)
        if features.mode == 'major':
            mood_indicators['happy'] += 0.3
            mood_indicators['energetic'] += 0.2
        else:
            mood_indicators['sad'] += 0.3
            mood_indicators['mysterious'] += 0.2
        
        # Tempo-based mood
        if features.tempo > 120:
            mood_indicators['energetic'] += 0.3
            mood_indicators['happy'] += 0.2
        elif features.tempo < 80:
            mood_indicators['calm'] += 0.3
            mood_indicators['sad'] += 0.2
        
        # Energy-based mood
        if energy_level == 'high':
            mood_indicators['energetic'] += 0.3
            mood_indicators['aggressive'] += 0.2
        elif energy_level == 'low':
            mood_indicators['calm'] += 0.3
        
        # Get dominant mood
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
        
        # Segment the audio based on spectral changes
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Use recurrence matrix to find repeated sections
        R = librosa.segment.recurrence_matrix(chroma)
        
        # Find segment boundaries
        boundaries = librosa.segment.agglomerative(chroma, k=8)
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)
        
        # Classify segments (simplified)
        segments = []
        for i in range(len(boundary_times) - 1):
            start_time = boundary_times[i]
            end_time = boundary_times[i + 1]
            duration = end_time - start_time
            
            # Simple classification based on position and duration
            if i == 0:
                segment_type = "intro"
            elif i == len(boundary_times) - 2:
                segment_type = "outro"
            elif duration > 30:
                segment_type = "verse" if i % 2 == 1 else "chorus"
            else:
                segment_type = "bridge"
            
            segments.append({
                'type': segment_type,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(duration)
            })
        
        return {
            'segments': segments,
            'total_segments': len(segments),
            'structure_complexity': 'simple' if len(segments) < 6 else 'complex'
        }
    
    def _identify_key_moments(self, y: np.ndarray, sr: int, features: AudioFeatures) -> List[Dict[str, Any]]:
        """Identify key moments for visual synchronization."""
        
        key_moments = []
        
        # Beat-synchronized moments
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Energy peaks
        rms = features.rms_energy[0]  # Take first channel
        rms_smooth = librosa.util.smooth(rms, length=5)
        
        # Find peaks in energy
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms_smooth, height=np.mean(rms_smooth) + np.std(rms_smooth))
        peak_times = librosa.frames_to_time(peaks, sr=sr)
        
        # Onset-based moments
        onset_times = features.onset_times
        
        # Combine and classify moments
        all_moments = []
        
        # Add beat moments (every 4th beat for emphasis)
        for i, beat_time in enumerate(beat_times[::4]):
            all_moments.append({
                'timestamp': float(beat_time),
                'type': 'beat_emphasis',
                'intensity': 0.6,
                'description': f'Strong beat at {beat_time:.1f}s'
            })
        
        # Add energy peaks
        for peak_time in peak_times:
            all_moments.append({
                'timestamp': float(peak_time),
                'type': 'energy_peak',
                'intensity': 0.8,
                'description': f'Energy peak at {peak_time:.1f}s'
            })
        
        # Add significant onsets
        for onset_time in onset_times[::3]:  # Every 3rd onset to avoid overcrowding
            all_moments.append({
                'timestamp': float(onset_time),
                'type': 'musical_onset',
                'intensity': 0.5,
                'description': f'Musical event at {onset_time:.1f}s'
            })
        
        # Sort by timestamp and remove duplicates
        all_moments.sort(key=lambda x: x['timestamp'])
        
        # Remove moments too close together (< 2 seconds)
        filtered_moments = []
        last_time = -2
        
        for moment in all_moments:
            if moment['timestamp'] - last_time >= 2:
                filtered_moments.append(moment)
                last_time = moment['timestamp']
        
        return filtered_moments[:20]  # Limit to 20 key moments
    
    def _generate_creative_insights(
        self, 
        genre: Dict[str, Any], 
        tempo: Dict[str, Any], 
        instruments: List[Dict[str, Any]], 
        mood: Dict[str, Any], 
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate creative insights for video production."""
        
        insights = {
            'visual_style_suggestions': [],
            'narrative_themes': [],
            'color_palette_suggestions': [],
            'camera_movement_suggestions': [],
            'editing_style_suggestions': []
        }
        
        # Genre-based suggestions
        primary_genre = genre['primary_genre']
        
        if primary_genre == 'electronic':
            insights['visual_style_suggestions'].extend([
                'Futuristic/cyberpunk aesthetics',
                'Neon lighting and digital effects',
                'Abstract geometric patterns'
            ])
            insights['color_palette_suggestions'].extend([
                'Electric blues and purples',
                'Neon greens and magentas',
                'High contrast black and white with color accents'
            ])
        
        elif primary_genre == 'rock':
            insights['visual_style_suggestions'].extend([
                'Gritty urban environments',
                'Performance-based footage',
                'High-energy concert atmosphere'
            ])
            insights['color_palette_suggestions'].extend([
                'Warm oranges and reds',
                'Desaturated earth tones',
                'High contrast lighting'
            ])
        
        # Tempo-based suggestions
        tempo_class = tempo['classification']
        
        if tempo_class in ['fast', 'very_fast']:
            insights['camera_movement_suggestions'].extend([
                'Quick cuts and rapid montages',
                'Handheld camera work',
                'Fast tracking shots'
            ])
            insights['editing_style_suggestions'].append('Fast-paced editing with beat synchronization')
        
        elif tempo_class in ['slow', 'very_slow']:
            insights['camera_movement_suggestions'].extend([
                'Smooth, flowing camera movements',
                'Long, contemplative shots',
                'Slow zoom and pan movements'
            ])
            insights['editing_style_suggestions'].append('Slow, deliberate cuts with longer shot durations')
        
        # Mood-based suggestions
        dominant_mood = mood['dominant_mood']
        
        if dominant_mood == 'happy':
            insights['narrative_themes'].extend([
                'Celebration and joy',
                'Success and achievement',
                'Love and relationships'
            ])
        
        elif dominant_mood == 'sad':
            insights['narrative_themes'].extend([
                'Loss and longing',
                'Introspection and reflection',
                'Overcoming adversity'
            ])
        
        elif dominant_mood == 'energetic':
            insights['narrative_themes'].extend([
                'Adventure and excitement',
                'Competition and challenge',
                'Freedom and rebellion'
            ])
        
        # Instrument-based suggestions
        prominent_instruments = [inst['instrument'] for inst in instruments[:3]]
        
        if 'guitar' in prominent_instruments:
            insights['visual_style_suggestions'].append('Band performance elements')
        
        if 'synthesizer' in prominent_instruments:
            insights['visual_style_suggestions'].append('Electronic/digital visual effects')
        
        if 'drums' in prominent_instruments:
            insights['editing_style_suggestions'].append('Rhythm-driven editing with drum synchronization')
        
        return insights


class LyricsIntelligence:
    """Advanced lyrics analysis for thematic and narrative insights."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common thematic keywords
        self.theme_keywords = {
            'love': ['love', 'heart', 'kiss', 'romance', 'together', 'forever', 'baby', 'darling'],
            'freedom': ['free', 'fly', 'escape', 'break', 'chains', 'liberty', 'open', 'sky'],
            'struggle': ['fight', 'battle', 'war', 'pain', 'hurt', 'struggle', 'overcome', 'survive'],
            'celebration': ['party', 'dance', 'celebrate', 'joy', 'happy', 'fun', 'tonight', 'music'],
            'journey': ['road', 'travel', 'journey', 'path', 'destination', 'adventure', 'explore'],
            'dreams': ['dream', 'hope', 'wish', 'believe', 'future', 'tomorrow', 'vision', 'imagine'],
            'rebellion': ['rebel', 'revolution', 'change', 'system', 'break', 'rules', 'authority'],
            'nostalgia': ['remember', 'past', 'yesterday', 'memories', 'time', 'ago', 'childhood'],
            'spirituality': ['god', 'heaven', 'soul', 'spirit', 'faith', 'pray', 'divine', 'sacred'],
            'nature': ['sun', 'moon', 'stars', 'ocean', 'mountain', 'forest', 'earth', 'sky']
        }
    
    def analyze_lyrics(self, lyrics: str) -> Dict[str, Any]:
        """Comprehensive lyrics analysis."""
        
        if not lyrics:
            return {'error': 'No lyrics provided'}
        
        # Clean and prepare lyrics
        clean_lyrics = self._clean_lyrics(lyrics)
        
        # Extract themes
        themes = self._extract_themes(clean_lyrics)
        
        # Analyze narrative structure
        narrative = self._analyze_narrative_structure(clean_lyrics)
        
        # Extract emotional content
        emotions = self._analyze_emotional_content(clean_lyrics)
        
        # Identify visual imagery
        imagery = self._extract_visual_imagery(clean_lyrics)
        
        # Generate creative suggestions
        creative_suggestions = self._generate_lyrical_creative_suggestions(
            themes, narrative, emotions, imagery
        )
        
        return {
            'themes': themes,
            'narrative_structure': narrative,
            'emotional_content': emotions,
            'visual_imagery': imagery,
            'creative_suggestions': creative_suggestions,
            'word_count': len(clean_lyrics.split()),
            'line_count': len(clean_lyrics.split('\n'))
        }
    
    def _clean_lyrics(self, lyrics: str) -> str:
        """Clean and normalize lyrics text."""
        # Remove extra whitespace and normalize
        lyrics = re.sub(r'\s+', ' ', lyrics.strip())
        
        # Remove common song structure markers
        lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove [Verse], [Chorus], etc.
        lyrics = re.sub(r'\(.*?\)', '', lyrics)  # Remove parenthetical notes
        
        return lyrics.lower()
    
    def _extract_themes(self, lyrics: str) -> Dict[str, Any]:
        """Extract thematic content from lyrics."""
        
        words = lyrics.split()
        theme_scores = {}
        
        for theme, keywords in self.theme_keywords.items():
            score = 0
            matched_words = []
            
            for keyword in keywords:
                count = lyrics.count(keyword)
                if count > 0:
                    score += count
                    matched_words.extend([keyword] * count)
            
            if score > 0:
                theme_scores[theme] = {
                    'score': score,
                    'matched_words': matched_words,
                    'prominence': 'high' if score > 3 else 'medium' if score > 1 else 'low'
                }
        
        # Sort themes by score
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return {
            'primary_themes': [theme[0] for theme in sorted_themes[:3]],
            'all_themes': dict(sorted_themes),
            'theme_diversity': len(theme_scores)
        }
    
    def _analyze_narrative_structure(self, lyrics: str) -> Dict[str, Any]:
        """Analyze narrative elements in lyrics."""
        
        lines = lyrics.split('\n')
        
        # Look for narrative indicators
        narrative_indicators = {
            'storytelling': ['story', 'once', 'remember', 'happened', 'told'],
            'dialogue': ['"', "'", 'said', 'told', 'asked'],
            'temporal': ['yesterday', 'today', 'tomorrow', 'now', 'then', 'when'],
            'characters': ['i', 'you', 'he', 'she', 'we', 'they'],
            'setting': ['here', 'there', 'home', 'city', 'street', 'room']
        }
        
        structure_scores = {}
        for category, indicators in narrative_indicators.items():
            score = sum(lyrics.count(indicator) for indicator in indicators)
            structure_scores[category] = score
        
        # Determine narrative type
        if structure_scores['storytelling'] > 2:
            narrative_type = 'story_driven'
        elif structure_scores['dialogue'] > 2:
            narrative_type = 'conversational'
        elif structure_scores['temporal'] > 3:
            narrative_type = 'temporal_journey'
        else:
            narrative_type = 'abstract_emotional'
        
        return {
            'narrative_type': narrative_type,
            'structure_elements': structure_scores,
            'has_clear_narrative': structure_scores['storytelling'] > 1,
            'character_focus': 'first_person' if lyrics.count('i') > lyrics.count('you') else 'second_person'
        }
    
    def _analyze_emotional_content(self, lyrics: str) -> Dict[str, Any]:
        """Analyze emotional content and intensity."""
        
        emotion_keywords = {
            'joy': ['happy', 'joy', 'smile', 'laugh', 'celebrate', 'amazing', 'wonderful'],
            'sadness': ['sad', 'cry', 'tears', 'lonely', 'empty', 'broken', 'hurt'],
            'anger': ['angry', 'mad', 'rage', 'hate', 'fury', 'fight', 'destroy'],
            'fear': ['afraid', 'scared', 'fear', 'terror', 'nightmare', 'panic'],
            'love': ['love', 'adore', 'cherish', 'devotion', 'passion', 'romance'],
            'hope': ['hope', 'believe', 'faith', 'trust', 'optimism', 'future'],
            'excitement': ['excited', 'thrilled', 'amazing', 'incredible', 'awesome']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(lyrics.count(keyword) for keyword in keywords)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Calculate emotional intensity
        total_emotional_words = sum(emotion_scores.values())
        total_words = len(lyrics.split())
        emotional_intensity = total_emotional_words / total_words if total_words > 0 else 0
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else 'neutral'
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotional_intensity': emotional_intensity,
            'emotion_scores': emotion_scores,
            'emotional_range': len(emotion_scores)
        }
    
    def _extract_visual_imagery(self, lyrics: str) -> Dict[str, Any]:
        """Extract visual imagery and descriptive elements."""
        
        visual_categories = {
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gold', 'silver'],
            'nature': ['sun', 'moon', 'stars', 'ocean', 'mountain', 'forest', 'sky', 'fire'],
            'urban': ['city', 'street', 'building', 'car', 'lights', 'crowd', 'noise'],
            'movement': ['run', 'fly', 'dance', 'jump', 'fall', 'rise', 'move', 'flow'],
            'textures': ['smooth', 'rough', 'soft', 'hard', 'cold', 'warm', 'bright', 'dark']
        }
        
        imagery_found = {}
        for category, words in visual_categories.items():
            found_words = [word for word in words if word in lyrics]
            if found_words:
                imagery_found[category] = found_words
        
        return {
            'visual_elements': imagery_found,
            'imagery_richness': len(imagery_found),
            'dominant_imagery': max(imagery_found.items(), key=lambda x: len(x[1]))[0] if imagery_found else 'abstract'
        }
    
    def _generate_lyrical_creative_suggestions(
        self, 
        themes: Dict[str, Any], 
        narrative: Dict[str, Any], 
        emotions: Dict[str, Any], 
        imagery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate creative suggestions based on lyrical analysis."""
        
        suggestions = {
            'visual_concepts': [],
            'narrative_approaches': [],
            'character_concepts': [],
            'setting_suggestions': []
        }
        
        # Theme-based visual concepts
        primary_themes = themes.get('primary_themes', [])
        
        for theme in primary_themes:
            if theme == 'love':
                suggestions['visual_concepts'].extend([
                    'Romantic couple storyline',
                    'Heart imagery and symbolism',
                    'Intimate, warm lighting'
                ])
            elif theme == 'freedom':
                suggestions['visual_concepts'].extend([
                    'Open landscapes and horizons',
                    'Breaking chains or barriers',
                    'Flying or soaring imagery'
                ])
            elif theme == 'journey':
                suggestions['visual_concepts'].extend([
                    'Road trip or travel narrative',
                    'Transformation sequence',
                    'Multiple locations/settings'
                ])
        
        # Narrative-based approaches
        narrative_type = narrative.get('narrative_type', 'abstract_emotional')
        
        if narrative_type == 'story_driven':
            suggestions['narrative_approaches'].append('Linear storytelling with clear beginning, middle, end')
        elif narrative_type == 'conversational':
            suggestions['narrative_approaches'].append('Dialogue-driven scenes between characters')
        elif narrative_type == 'temporal_journey':
            suggestions['narrative_approaches'].append('Time-based narrative showing past, present, future')
        
        # Emotion-based character concepts
        dominant_emotion = emotions.get('dominant_emotion', 'neutral')
        
        if dominant_emotion == 'joy':
            suggestions['character_concepts'].append('Celebratory, energetic protagonist')
        elif dominant_emotion == 'sadness':
            suggestions['character_concepts'].append('Melancholic character seeking resolution')
        elif dominant_emotion == 'love':
            suggestions['character_concepts'].append('Romantic leads with emotional connection')
        
        # Imagery-based settings
        dominant_imagery = imagery.get('dominant_imagery', 'abstract')
        
        if dominant_imagery == 'nature':
            suggestions['setting_suggestions'].extend([
                'Natural outdoor environments',
                'Seasonal changes and weather',
                'Wildlife and natural phenomena'
            ])
        elif dominant_imagery == 'urban':
            suggestions['setting_suggestions'].extend([
                'City streets and architecture',
                'Nightlife and urban energy',
                'Modern, contemporary settings'
            ])
        
        return suggestions


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    audio_analyzer = AudioIntelligence()
    lyrics_analyzer = LyricsIntelligence()
    
    # Test with sample data
    sample_lyrics = """
    I'm running through the city lights tonight
    Feeling free, feeling alive
    The music's pumping, heart is beating
    This is our time, this is our night
    """
    
    lyrics_analysis = lyrics_analyzer.analyze_lyrics(sample_lyrics)
    print("Lyrics Analysis:")
    print(json.dumps(lyrics_analysis, indent=2))