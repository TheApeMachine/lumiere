"""
Example usage of the Agent-based Filmmaking Team.

This demonstrates how the multi-agent system creates dynamic, story-driven
music video concepts that go far beyond simple image variations.
"""

import asyncio
import json
import logging
from pathlib import Path

from filmmaking_team import FilmmakingTeam
from audio_intelligence import AudioIntelligence, LyricsIntelligence


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_filmmaking_team():
    """Demonstrate the complete filmmaking team workflow."""
    
    print("🎬 LUMIERE AI - Agent-Based Filmmaking Team Demo")
    print("=" * 60)
    
    # Initialize the team
    team = FilmmakingTeam()
    audio_intel = AudioIntelligence()
    lyrics_intel = LyricsIntelligence()
    
    # Example 1: Electronic Dance Music
    print("\n🎵 Example 1: Electronic Dance Track")
    print("-" * 40)
    
    # Simulate audio analysis results
    edm_analysis = {
        'genre': {'primary_genre': 'electronic', 'confidence': 0.85},
        'tempo': {'bpm': 128, 'classification': 'fast', 'stability': 'stable'},
        'instruments': [
            {'instrument': 'synthesizer', 'confidence': 0.9, 'prominence': 'high'},
            {'instrument': 'drums', 'confidence': 0.8, 'prominence': 'high'},
            {'instrument': 'bass', 'confidence': 0.7, 'prominence': 'medium'}
        ],
        'mood_and_energy': {
            'energy_level': 'high',
            'dominant_mood': 'energetic',
            'mood_confidence': 0.8
        },
        'key_moments': [
            {'timestamp': 0.0, 'type': 'intro', 'intensity': 0.3, 'description': 'Atmospheric buildup'},
            {'timestamp': 32.0, 'type': 'drop', 'intensity': 0.9, 'description': 'Main drop with heavy bass'},
            {'timestamp': 64.0, 'type': 'breakdown', 'intensity': 0.4, 'description': 'Melodic breakdown'},
            {'timestamp': 96.0, 'type': 'climax', 'intensity': 1.0, 'description': 'Final explosive climax'}
        ]
    }
    
    edm_lyrics = """
    Feel the beat inside your soul
    Let the music take control
    We're flying high above the clouds
    Dancing with the stars so loud
    
    This is our moment, this is our time
    Electric energy, rhythm and rhyme
    Break free from gravity's hold
    Let your story now unfold
    """
    
    # Analyze lyrics
    lyrics_analysis = lyrics_intel.analyze_lyrics(edm_lyrics)
    print(f"📝 Lyrics Analysis:")
    print(f"   Primary themes: {lyrics_analysis['themes']['primary_themes']}")
    print(f"   Dominant emotion: {lyrics_analysis['emotional_content']['dominant_emotion']}")
    print(f"   Narrative type: {lyrics_analysis['narrative_structure']['narrative_type']}")
    
    # Create concept with filmmaking team
    try:
        print("\n🎭 Creating concept with filmmaking team...")
        
        # Simulate the agent workflow
        concept_result = await simulate_agent_workflow(
            "edm_track.mp3", 
            "Create something futuristic and high-energy",
            edm_lyrics,
            edm_analysis,
            lyrics_analysis
        )
        
        print(f"✅ Concept created successfully!")
        print(f"📖 Story: {concept_result['story_summary']}")
        print(f"🎥 Visual Style: {concept_result['visual_style']}")
        print(f"🎨 Color Palette: {', '.join(concept_result['color_palette'])}")
        print(f"📹 Total Scenes: {len(concept_result['scenes'])}")
        
        # Show first few scenes
        print(f"\n🎬 Scene Breakdown (first 3 scenes):")
        for i, scene in enumerate(concept_result['scenes'][:3]):
            print(f"   Scene {i+1} ({scene['timestamp']}s): {scene['description']}")
            print(f"      Visual: {scene['visual_prompt'][:100]}...")
        
    except Exception as e:
        logger.error(f"Error in EDM example: {e}")
    
    print("\n" + "=" * 60)
    
    # Example 2: Indie Rock Ballad
    print("\n🎸 Example 2: Indie Rock Ballad")
    print("-" * 40)
    
    rock_analysis = {
        'genre': {'primary_genre': 'rock', 'confidence': 0.78},
        'tempo': {'bpm': 85, 'classification': 'slow', 'stability': 'variable'},
        'instruments': [
            {'instrument': 'guitar', 'confidence': 0.9, 'prominence': 'high'},
            {'instrument': 'vocals', 'confidence': 0.85, 'prominence': 'high'},
            {'instrument': 'drums', 'confidence': 0.6, 'prominence': 'medium'},
            {'instrument': 'bass', 'confidence': 0.5, 'prominence': 'low'}
        ],
        'mood_and_energy': {
            'energy_level': 'medium',
            'dominant_mood': 'melancholic',
            'mood_confidence': 0.75
        },
        'key_moments': [
            {'timestamp': 0.0, 'type': 'intro', 'intensity': 0.2, 'description': 'Gentle guitar intro'},
            {'timestamp': 45.0, 'type': 'verse_climax', 'intensity': 0.6, 'description': 'Emotional vocal peak'},
            {'timestamp': 90.0, 'type': 'bridge', 'intensity': 0.8, 'description': 'Instrumental bridge with guitar solo'},
            {'timestamp': 135.0, 'type': 'outro', 'intensity': 0.3, 'description': 'Fade out with vocals'}
        ]
    }
    
    rock_lyrics = """
    Walking down this empty street
    Memories beneath my feet
    Every step takes me away
    From the words I couldn't say
    
    If I could turn back time
    Make your heart align with mine
    But the past is set in stone
    Now I'm learning to be alone
    
    The city lights blur through my tears
    As I face my deepest fears
    Tomorrow's just another day
    To find my own way
    """
    
    rock_lyrics_analysis = lyrics_intel.analyze_lyrics(rock_lyrics)
    print(f"📝 Lyrics Analysis:")
    print(f"   Primary themes: {rock_lyrics_analysis['themes']['primary_themes']}")
    print(f"   Dominant emotion: {rock_lyrics_analysis['emotional_content']['dominant_emotion']}")
    print(f"   Visual imagery: {rock_lyrics_analysis['visual_imagery']['dominant_imagery']}")
    
    try:
        print("\n🎭 Creating concept with filmmaking team...")
        
        rock_concept = await simulate_agent_workflow(
            "indie_ballad.mp3",
            "Tell a story of loss and self-discovery",
            rock_lyrics,
            rock_analysis,
            rock_lyrics_analysis
        )
        
        print(f"✅ Concept created successfully!")
        print(f"📖 Story: {rock_concept['story_summary']}")
        print(f"🎥 Visual Style: {rock_concept['visual_style']}")
        print(f"🎨 Mood: {rock_concept['mood']}")
        print(f"📹 Total Scenes: {len(rock_concept['scenes'])}")
        
    except Exception as e:
        logger.error(f"Error in rock example: {e}")
    
    print("\n🎉 Demo completed! The agent-based system creates unique,")
    print("   story-driven concepts instead of just image variations.")


async def simulate_agent_workflow(audio_file, user_prompt, lyrics, audio_analysis, lyrics_analysis):
    """
    Simulate the agent workflow to show how different agents collaborate.
    In a real implementation, this would use the actual Agent SDK.
    """
    
    # Simulate Music Analyst Agent
    print("   🎵 Music Analyst: Analyzing audio characteristics...")
    await asyncio.sleep(0.5)  # Simulate processing time
    
    music_insights = {
        'genre': audio_analysis['genre']['primary_genre'],
        'energy': audio_analysis['mood_and_energy']['energy_level'],
        'tempo': audio_analysis['tempo']['classification'],
        'instruments': [inst['instrument'] for inst in audio_analysis['instruments'][:3]],
        'key_moments': audio_analysis['key_moments']
    }
    
    # Simulate Writer Agent
    print("   ✍️  Writer: Creating narrative treatment...")
    await asyncio.sleep(1.0)
    
    # Generate story based on analysis
    if lyrics_analysis['themes']['primary_themes']:
        primary_theme = lyrics_analysis['themes']['primary_themes'][0]
    else:
        primary_theme = 'journey'
    
    if primary_theme == 'freedom':
        story = "A protagonist breaks free from constraints and discovers their true potential"
        characters = ["Free-spirited protagonist", "Symbolic representations of constraints"]
    elif primary_theme == 'love':
        story = "A romantic journey through the highs and lows of deep connection"
        characters = ["Romantic leads", "Supporting ensemble representing different aspects of love"]
    elif primary_theme == 'struggle':
        story = "An individual overcomes personal challenges to achieve transformation"
        characters = ["Determined protagonist", "Symbolic obstacles and allies"]
    else:
        story = "A character's emotional and physical journey of self-discovery"
        characters = ["Introspective protagonist", "Environmental elements as characters"]
    
    # Simulate Director of Photography Agent
    print("   🎥 Director of Photography: Planning cinematography...")
    await asyncio.sleep(0.8)
    
    # Visual style based on genre and mood
    if music_insights['genre'] == 'electronic':
        visual_style = "Futuristic cyberpunk with neon lighting and digital effects"
        color_palette = ["Electric blue", "Neon purple", "Chrome silver", "Deep black"]
        camera_style = "Dynamic tracking shots with quick cuts synchronized to beats"
    elif music_insights['genre'] == 'rock':
        visual_style = "Gritty urban realism with warm, cinematic lighting"
        color_palette = ["Warm amber", "Deep orange", "Muted gold", "Shadow black"]
        camera_style = "Handheld intimacy with slow, emotional movements"
    else:
        visual_style = "Contemporary cinematic with balanced lighting"
        color_palette = ["Natural tones", "Soft blues", "Warm whites"]
        camera_style = "Smooth, flowing camera movements"
    
    # Simulate Director Agent
    print("   🎬 Director: Reviewing and approving concept...")
    await asyncio.sleep(0.7)
    
    # Generate scenes based on key moments
    scenes = []
    for i, moment in enumerate(music_insights['key_moments']):
        scene_description = f"{moment['description']} - {story} (Scene {i+1})"
        
        visual_prompt = f"{scene_description}. {visual_style}. {camera_style}. " \
                       f"Color palette: {', '.join(color_palette[:2])}. " \
                       f"Cinematic quality, professional music video production, " \
                       f"emotional intensity: {moment['intensity']}"
        
        scenes.append({
            'timestamp': moment['timestamp'],
            'duration': 8.0,  # Default scene duration
            'description': scene_description,
            'visual_prompt': visual_prompt,
            'intensity': moment['intensity'],
            'camera_movement': camera_style.split()[0].lower()
        })
    
    print("   ✅ Director: Concept approved!")
    
    return {
        'story_summary': story,
        'characters': characters,
        'visual_style': visual_style,
        'color_palette': color_palette,
        'camera_style': camera_style,
        'mood': lyrics_analysis['emotional_content']['dominant_emotion'],
        'scenes': scenes,
        'total_duration': max(scene['timestamp'] for scene in scenes) + 10
    }


def compare_with_traditional_approach():
    """Show the difference between traditional and agent-based approaches."""
    
    print("\n🔄 COMPARISON: Traditional vs Agent-Based Approach")
    print("=" * 60)
    
    print("\n❌ Traditional Approach:")
    print("   • Generate 6 similar images from same seed prompt")
    print("   • No story or narrative structure")
    print("   • Limited creative variation")
    print("   • No consideration of music characteristics")
    print("   • Static, repetitive results")
    
    print("\n✅ Agent-Based Filmmaking Team:")
    print("   • Music Analyst extracts genre, tempo, instruments, mood")
    print("   • Writer creates compelling narrative based on music + lyrics")
    print("   • DoP plans cinematography matching the story and music")
    print("   • Director reviews and requests revisions for quality")
    print("   • Each scene has unique, story-driven prompts")
    print("   • Dynamic collaboration creates fresh, engaging concepts")
    print("   • Considers musical structure for visual synchronization")
    
    print("\n🎯 Result: Instead of 6 variations of the same image,")
    print("   you get a complete, cohesive music video concept with")
    print("   narrative arc, character development, and visual storytelling!")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_filmmaking_team())
    
    # Show comparison
    compare_with_traditional_approach()
    
    print("\n🚀 Ready to revolutionize music video creation!")
    print("   Start the creative director service to begin using the")
    print("   agent-based filmmaking team in your Lumiere pipeline.")