"""
Creative Director Service - Agent-based filmmaking team server.

This service provides an API for the agent-based filmmaking team that creates
dynamic, story-driven music video concepts.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify
from flask_cors import CORS

from filmmaking_team import FilmmakingTeam, AudioAnalyzer, LyricsProcessor
from audio_intelligence import AudioIntelligence, LyricsIntelligence
from models import (
    SeedPromptsRequest,
    SeedPromptsResponse,
    VideoPromptsRequest,
    VideoPromptsResponse,
    Concept,
    Character,
    StoryPlan,
    Scene,
    StoryBeat,
    SeedImagePrompt,
    VideoPromptSegment,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize filmmaking team and intelligence modules
filmmaking_team = FilmmakingTeam()
audio_intelligence = AudioIntelligence()
lyrics_intelligence = LyricsIntelligence()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "creative-director"})


@app.route('/healthz', methods=['GET'])
def healthz_check():
    """Kubernetes-style health check."""
    return jsonify({"status": "ok"})


def _infer_concept(concept: Concept, audio_summary: Dict[str, Any]) -> Concept:
    """Fill missing concept fields based on audio summary or defaults."""
    theme = concept.theme or _guess_theme(audio_summary)
    mood = concept.mood or _guess_mood(audio_summary)
    visual_style = concept.visual_style or _guess_style(audio_summary)
    return Concept(
        title=concept.title,
        theme=theme,
        mood=mood,
        visual_style=visual_style,
        characters=concept.characters or [],
    )


def _guess_theme(audio_summary: Dict[str, Any]) -> str:
    return "transformation"


def _guess_mood(audio_summary: Dict[str, Any]) -> str:
    bpm = audio_summary.get("bpm") or audio_summary.get("tempo")
    try:
        bpm_val = float(bpm) if bpm is not None else 100.0
    except Exception:
        bpm_val = 100.0
    return "brooding" if bpm_val < 90 else ("uplifting" if bpm_val > 130 else "driving")


def _guess_style(audio_summary: Dict[str, Any]) -> str:
    return "cinematic music video, anamorphic, rich contrast"


def _build_story_plan(audio_summary: Dict[str, Any], concept_final: Concept) -> StoryPlan:
    beats: List[float] = audio_summary.get("beats") or audio_summary.get("beat_times") or []
    if not beats:
        # Fallback: 7 evenly spaced beats over 90s
        beats = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]

    # Create scenes by grouping beats into 3 acts
    n = len(beats)
    cut1 = max(1, n // 3)
    cut2 = max(cut1 + 1, (2 * n) // 3)
    act_splits = [(0, cut1), (cut1, cut2), (cut2, n)]

    locations = [
        "alley in the rain",
        "rooftop at dusk",
        "neon market",
        "subway platform",
        "riverfront at night",
        "bridge in fog",
        "city overlook",
    ]
    times = ["dawn", "day", "dusk", "night"]
    palettes = [
        ["cool teal", "neon magenta"],
        ["amber", "deep blue"],
        ["violet", "electric cyan"],
    ]

    scenes: List[Scene] = []
    scene_index = 0
    for act_idx, (s, e) in enumerate(act_splits):
        for i in range(s, e - 1):
            start_bt = float(beats[i])
            end_bt = float(beats[i + 1])
            beat = StoryBeat(
                beat_index=i,
                label=["setup", "rising", "climax"][min(act_idx, 2)],
                start_beat=start_bt,
                end_beat=end_bt,
            )

            loc = locations[min(scene_index, len(locations) - 1)]
            tod = times[(scene_index + act_idx) % len(times)]
            pal = palettes[min(act_idx, len(palettes) - 1)]
            chars = [c.id for c in concept_final.characters][:2]

            scenes.append(
                Scene(
                    scene_index=scene_index,
                    location=loc,
                    time_of_day=tod,
                    characters_on_stage=chars,
                    palette=pal,
                    beats=[beat],
                )
            )
            scene_index += 1

    logline = f"{concept_final.title}: a {concept_final.mood} journey of {concept_final.theme}."
    return StoryPlan(logline=logline, acts=3, scenes=scenes)


def _prompts_from_story(story: StoryPlan, concept_final: Concept, num_variations: int) -> List[SeedImagePrompt]:
    prompts: List[SeedImagePrompt] = []
    style_tags = [concept_final.visual_style or "cinematic"]
    for sc in story.scenes:
        for bt in sc.beats:
            text = (
                f"{sc.location}, {sc.time_of_day}, {concept_final.mood} mood, "
                f"{concept_final.visual_style}; characters: {', '.join(sc.characters_on_stage)}"
            )
            ref_imgs = []
            for char_id in sc.characters_on_stage:
                ch = next((c for c in concept_final.characters if c.id == char_id), None)
                if ch and ch.reference_images:
                    ref_imgs.append(ch.reference_images[0])

            prompts.append(
                SeedImagePrompt(
                    prompt=text,
                    negative_prompt="low-res, blurry, duplicate",
                    start_beat=bt.start_beat,
                    end_beat=bt.end_beat,
                    seed=None,
                    style_tags=style_tags,
                    characters=sc.characters_on_stage,
                    reference_images=ref_imgs,
                )
            )
            if len(prompts) >= num_variations:
                return prompts
    return prompts


def _segments_from_story(story: StoryPlan, concept_final: Concept) -> List[VideoPromptSegment]:
    segments: List[VideoPromptSegment] = []
    for sc in story.scenes:
        for bt in sc.beats:
            prompt = (
                f"Tracking shot through {sc.location} at {sc.time_of_day}, {concept_final.visual_style}; "
                f"characters {', '.join(sc.characters_on_stage)} advance goals"
            )
            ref_imgs = []
            for char_id in sc.characters_on_stage:
                ch = next((c for c in concept_final.characters if c.id == char_id), None)
                if ch and ch.reference_images:
                    ref_imgs.append(ch.reference_images[0])
            segments.append(
                VideoPromptSegment(
                    start_beat=bt.start_beat,
                    end_beat=bt.end_beat,
                    prompt=prompt,
                    negative_prompt="glitches, artifacts",
                    motion_notes="motivated camera, evolving blocking",
                    transition="musical cut",
                    characters_on_screen=sc.characters_on_stage,
                    reference_images=ref_imgs,
                    scene_index=sc.scene_index,
                    beat_index=bt.beat_index,
                )
            )
    return segments


def _write_snapshot(project_id: str, kind: str, payload: Dict[str, Any]) -> None:
    try:
        base = os.path.join(os.path.dirname(__file__), "snapshots")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"{project_id}-{kind}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write snapshot: {e}")


@app.route('/analyze-music', methods=['POST'])
def analyze_music():
    """Analyze music file for creative insights."""
    try:
        data = request.get_json()
        audio_file = data.get('audio_file')
        lyrics = data.get('lyrics')

        if not audio_file:
            return jsonify({"error": "audio_file is required"}), 400

        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Use advanced audio intelligence
            analysis = audio_intelligence.analyze_audio_file(audio_file)

            # Process lyrics if provided
            lyrics_analysis = None
            if lyrics:
                lyrics_analysis = lyrics_intelligence.analyze_lyrics(lyrics)

            return jsonify({
                "music_analysis": analysis,
                "lyrics_analysis": lyrics_analysis
            })

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error analyzing music: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/create-concept', methods=['POST'])
def create_concept():
    """Create a complete music video concept using the filmmaking team."""
    try:
        data = request.get_json()
        audio_file = data.get('audio_file')
        user_prompt = data.get('prompt', '')
        lyrics = data.get('lyrics')

        if not audio_file:
            return jsonify({"error": "audio_file is required"}), 400

        logger.info(f"Creating concept for audio file: {audio_file}")

        # Run async concept creation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Create concept using filmmaking team
            director_vision = loop.run_until_complete(
                filmmaking_team.create_music_video_concept(
                    audio_file_path=audio_file,
                    user_prompt=user_prompt,
                    lyrics=lyrics
                )
            )

            # Generate scene prompts
            scene_prompts = loop.run_until_complete(
                filmmaking_team.generate_scene_prompts(director_vision)
            )

            response = {
                "director_vision": director_vision.dict() if hasattr(director_vision, 'dict') else str(director_vision),
                "scene_prompts": scene_prompts,
                "total_scenes": len(scene_prompts)
            }

            logger.info(f"Concept created successfully with {len(scene_prompts)} scenes")
            return jsonify(response)

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error creating concept: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/seed_prompts', methods=['POST'])
def v1_seed_prompts():
    """Generate story-first seed image prompts with character continuity."""
    try:
        req = SeedPromptsRequest(**(request.get_json() or {}))

        concept_final = _infer_concept(req.concept, req.audio_summary)
        story = _build_story_plan(req.audio_summary, concept_final)
        prompts = _prompts_from_story(story, concept_final, req.num_variations)

        resp = SeedPromptsResponse(
            concept_final=concept_final,
            story_plan=story,
            prompts=prompts,
        )

        payload = json.loads(resp.model_dump_json())
        _write_snapshot(req.project_id, "seed_prompts", payload)
        return jsonify(payload)
    except Exception as e:
        logger.error(f"/v1/seed_prompts error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/video_prompts', methods=['POST'])
def v1_video_prompts():
    """Generate video prompt segments aligned to beats and evolving scenes."""
    try:
        req = VideoPromptsRequest(**(request.get_json() or {}))

        concept_final = _infer_concept(req.concept, req.audio_summary)
        story = _build_story_plan(req.audio_summary, concept_final)
        segments = _segments_from_story(story, concept_final)

        resp = VideoPromptsResponse(
            concept_final=concept_final,
            story_plan=story,
            segments=segments,
        )
        payload = json.loads(resp.model_dump_json())
        _write_snapshot(req.project_id, "video_prompts", payload)
        return jsonify(payload)
    except Exception as e:
        logger.error(f"/v1/video_prompts error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate-scene-prompts', methods=['POST'])
def generate_scene_prompts():
    """Generate detailed prompts for specific scenes."""
    try:
        data = request.get_json()
        director_vision_data = data.get('director_vision')

        if not director_vision_data:
            return jsonify({"error": "director_vision is required"}), 400

        # Convert dict back to DirectorVision object if needed
        # This is a simplified approach - in production you'd want proper serialization

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # For now, create mock scene prompts based on the vision data
            scene_prompts = []

            # Extract key moments from the vision
            if isinstance(director_vision_data, dict):
                scene_breakdown = director_vision_data.get('scene_breakdown', [])

                for i, scene in enumerate(scene_breakdown):
                    prompt = {
                        "scene_id": i,
                        "timestamp": scene.get("timestamp", i * 10),
                        "duration": scene.get("duration", 5.0),
                        "description": scene.get("description", f"Scene {i+1}"),
                        "detailed_prompt": f"Cinematic scene: {scene.get('description', f'Scene {i+1}')}. Professional music video quality.",
                        "visual_style": director_vision_data.get('approved_cinematography', {}).get('visual_style', 'modern'),
                        "mood": scene.get("mood", "dynamic")
                    }
                    scene_prompts.append(prompt)

            return jsonify({
                "scene_prompts": scene_prompts,
                "total_scenes": len(scene_prompts)
            })

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error generating scene prompts: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/revise-concept', methods=['POST'])
def revise_concept():
    """Request revisions to an existing concept."""
    try:
        data = request.get_json()
        current_concept = data.get('current_concept')
        revision_notes = data.get('revision_notes')
        target_agent = data.get('target_agent', 'writer')  # writer, dop, or music_analyst

        if not current_concept or not revision_notes:
            return jsonify({"error": "current_concept and revision_notes are required"}), 400

        logger.info(f"Requesting revision from {target_agent}: {revision_notes}")

        # In a full implementation, this would route the revision request
        # to the appropriate agent and regenerate the concept

        return jsonify({
            "status": "revision_requested",
            "target_agent": target_agent,
            "revision_notes": revision_notes,
            "message": "Revision request processed. Use /create-concept to generate updated concept."
        })

    except Exception as e:
        logger.error(f"Error processing revision: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/agent-status', methods=['GET'])
def agent_status():
    """Get status of all agents in the filmmaking team."""
    try:
        return jsonify({
            "agents": {
                "music_analyst": {"status": "ready", "role": "Analyzes audio tracks for creative insights"},
                "writer": {"status": "ready", "role": "Creates narrative treatments and stories"},
                "dop": {"status": "ready", "role": "Plans cinematography and visual style"},
                "director": {"status": "ready", "role": "Reviews and approves final concepts"}
            },
            "team_status": "ready"
        })

    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/creative-brief', methods=['POST'])
def create_creative_brief():
    """Create a creative brief based on user input and music analysis."""
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        music_analysis = data.get('music_analysis')
        target_audience = data.get('target_audience', 'general')
        budget_level = data.get('budget_level', 'medium')

        # Generate creative brief
        brief = {
            "project_overview": f"Music video concept based on user direction: {user_input}",
            "target_audience": target_audience,
            "budget_considerations": budget_level,
            "creative_direction": user_input,
            "music_insights": music_analysis,
            "deliverables": [
                "Story treatment",
                "Cinematography plan",
                "Scene breakdown",
                "Visual prompts for generation"
            ],
            "timeline": "Concept development: immediate, Production planning: varies"
        }

        return jsonify({"creative_brief": brief})

    except Exception as e:
        logger.error(f"Error creating creative brief: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting Creative Director service on port {port}")
    logger.info("Filmmaking team agents initialized and ready")

    app.run(host='0.0.0.0', port=port, debug=debug)