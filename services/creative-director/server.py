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

# Try to import local LLM support
try:
    from local_agents import LocalLLM
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    LocalLLM = None
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
audio_intelligence = AudioIntelligence()
lyrics_intelligence = LyricsIntelligence()

# Initialize local LLM lazily (in background thread to avoid blocking server startup)
llm = None
llm_initializing = False
llm_init_error = None

def initialize_llm_async():
    """Initialize LLM in background thread to avoid blocking server startup."""
    global llm, llm_initializing, llm_init_error
    
    if not LOCAL_LLM_AVAILABLE:
        return
    
    if llm_initializing or llm is not None:
        return
    
    llm_initializing = True
    
    try:
        # Set defaults for all environment variables (will auto-download models if needed)
        # Default to Mixtral 8x7B for better quality (fits comfortably in 128GB unified memory)
        # Use transformers by default - more reliable and simpler than GGUF
        model_path = os.environ.get('LOCAL_LLM_MODEL_PATH') or os.environ.get('LOCAL_LLM_MODEL') or None
        model_name = os.environ.get('LOCAL_LLM_MODEL_NAME') or os.environ.get('LOCAL_LLM_MODEL') or 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        use_gguf = os.environ.get('LOCAL_LLM_USE_GGUF', 'false').lower() == 'true'  # Default to transformers
        
        logger.info(f"Initializing local LLM in background: model_name={model_name}, use_gguf={use_gguf}")
        logger.info("Using transformers library (models download automatically like normal)")
        logger.info("This may take several minutes if downloading the model for the first time...")
        
        llm = LocalLLM(model_path=model_path, model_name=model_name, use_gguf=use_gguf, auto_download=True)
        logger.info("Local LLM initialized successfully")
        
        # Update filmmaking team with the new LLM
        global filmmaking_team
        filmmaking_team = FilmmakingTeam(audio_intelligence=audio_intelligence, llm=llm)
        logger.info("Filmmaking team updated with local LLM")
        
    except Exception as e:
        logger.error(f"Failed to initialize local LLM: {e}", exc_info=True)
        llm_init_error = str(e)
        logger.warning("Falling back to OpenAI Agents SDK (if available)")
        
        # Check if OpenAI Agents SDK is available as fallback
        try:
            import agents
            logger.info("OpenAI Agents SDK is available as fallback")
        except ImportError:
            logger.error("Neither local LLM nor OpenAI Agents SDK is available!")
            logger.error("Please install transformers: pip install transformers torch")
            logger.error("Or set OPENAI_API_KEY environment variable for OpenAI Agents SDK")
    finally:
        llm_initializing = False

# Start LLM initialization in background thread
import threading
llm_init_thread = threading.Thread(target=initialize_llm_async, daemon=True)
llm_init_thread.start()

# Initialize filmmaking team without LLM initially (will be updated when LLM loads)
filmmaking_team = FilmmakingTeam(audio_intelligence=audio_intelligence, llm=None)

def wait_for_llm(timeout=600):
    """Wait for LLM to be ready, with timeout (default 10 minutes for large model downloads)."""
    import time
    start_time = time.time()
    while llm_initializing and (time.time() - start_time) < timeout:
        time.sleep(1)
    return llm is not None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = "healthy"
    if llm_initializing:
        status = "initializing"
    elif llm is None and llm_init_error:
        status = "degraded"  # Service running but LLM failed
    
    return jsonify({
        "status": status,
        "service": "creative-director",
        "llm_ready": llm is not None,
        "llm_initializing": llm_initializing,
        "llm_error": llm_init_error if llm is None else None
    })


@app.route('/healthz', methods=['GET'])
def healthz_check():
    """Kubernetes-style health check."""
    # Return 200 even if LLM is still initializing - service is up and can handle requests
    # The LLM will be ready when needed (lazy initialization)
    return jsonify({"status": "ok"}), 200


def _infer_concept(concept: Concept, audio_summary: Dict[str, Any]) -> Concept:
    """Fill missing concept fields using simple heuristics and defaults.

    Ensures `mood`, `visual_style`, and at least one character are present so downstream
    prompt/story builders can rely on them.
    """
    title = concept.title
    theme = concept.theme or (audio_summary.get("theme") if isinstance(audio_summary, dict) else None) or "change"
    # Derive a basic mood from audio summary if available
    mood = concept.mood or (audio_summary.get("mood") if isinstance(audio_summary, dict) else None)
    if not mood:
        bpm = audio_summary.get("bpm") if isinstance(audio_summary, dict) else None
        try:
            mood = "energetic" if bpm and float(bpm) >= 110 else "contemplative"
        except Exception:
            mood = "dynamic"

    visual_style = concept.visual_style or "cinematic"

    characters: List[Character] = concept.characters or []
    if not characters:
        characters = [
            Character(id="lead", name="Protagonist", role="lead", descriptors=["distinctive", "memorable"], reference_images=[])
        ]

    return Concept(
        title=title,
        theme=theme,
        mood=mood,
        visual_style=visual_style,
        characters=characters,
    )


def _build_story_plan(audio_summary: Dict[str, Any], concept_final: Concept) -> StoryPlan:
    """Construct a minimal but structured StoryPlan from beats and concept.

    Groups beats into simple scenes (2 beats per scene) with placeholder locations
    and maps characters by id.
    """
    beats_raw: List[float] = []
    if isinstance(audio_summary, dict):
        beats_raw = audio_summary.get("beats") or audio_summary.get("beat_times") or []

    # Ensure we have at least a handful of beat boundaries
    if not beats_raw or len(beats_raw) < 2:
        # Create a fallback beat grid (0, 10, 20, ...)
        beats_raw = [float(i * 10.0) for i in range(0, 8)]

    # Create StoryBeats
    story_beats: List[StoryBeat] = []
    for i in range(len(beats_raw) - 1):
        story_beats.append(
            StoryBeat(
                beat_index=i,
                label=f"Beat {i+1}",
                start_beat=float(beats_raw[i]),
                end_beat=float(beats_raw[i + 1]),
            )
        )

    # Group beats into scenes: 2 beats per scene
    scenes: List[Scene] = []
    characters_on_stage = [c.id for c in (concept_final.characters or [])] or ["lead"]
    for s_idx in range(0, len(story_beats), 2):
        scene_beats = story_beats[s_idx:s_idx + 2]
        if not scene_beats:
            continue
        location = "urban exterior" if (s_idx // 2) % 2 == 0 else "interior studio"
        time_of_day = "night" if concept_final.mood and concept_final.mood.lower() in {"moody", "dark"} else "day"
        scenes.append(
            Scene(
                scene_index=len(scenes),
                location=location,
                time_of_day=time_of_day,
                characters_on_stage=characters_on_stage[:2],
                palette=[],
                beats=scene_beats,
            )
        )

    logline = f"A {concept_final.mood or 'dynamic'} {concept_final.theme or 'theme'} journey inspired by the music."
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

        # If we have an audio file, use the agent team to create director vision and scene prompts.
        if req.audio_file:
            # Wait for LLM to be ready (with timeout - 10 minutes for large model downloads)
            if llm_initializing:
                logger.info("LLM is still initializing, waiting...")
                if not wait_for_llm(timeout=600):
                    return jsonify({
                        "error": "LLM initialization timed out. Please check logs and try again.",
                        "llm_initializing": True
                    }), 503
            
            if llm is None and LOCAL_LLM_AVAILABLE:
                return jsonify({
                    "error": "LLM not available. Check logs for initialization errors.",
                    "llm_error": llm_init_error
                }), 503
            
            logger.info("\n" + "=" * 80)
            logger.info("[SEED_PROMPTS ENDPOINT] Starting filmmaking team pipeline")
            logger.info(f"Audio file: {req.audio_file}")
            logger.info(f"Project ID: {req.project_id}")
            logger.info(f"Number of variations requested: {req.num_variations}")
            logger.info("=" * 80)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                logger.info("\n[CREATING MUSIC VIDEO CONCEPT...]")
                director_vision = loop.run_until_complete(
                    filmmaking_team.create_music_video_concept(
                        audio_file_path=req.audio_file,
                        user_prompt=concept_final.title,
                        lyrics=None,
                    )
                )
                logger.info(f"\n[CONCEPT CREATED] Type: {type(director_vision)}")
                logger.info(f"Director vision has scene_breakdown: {hasattr(director_vision, 'scene_breakdown')}")

                logger.info("\n[GENERATING SCENE PROMPTS...]")
                scene_prompts = loop.run_until_complete(
                    filmmaking_team.generate_scene_prompts(director_vision)
                )
                logger.info(f"\n[SCENE PROMPTS GENERATED] Total: {len(scene_prompts)}")
                for i, sp in enumerate(scene_prompts[:3]):
                    logger.info(f"  Scene {i+1}: {sp.get('detailed_prompt', 'N/A')[:100]}...")
            finally:
                loop.close()

            # Map scene prompts onto beat intervals to produce seed prompts
            logger.info("\n[MAPPING SCENE PROMPTS TO SEED IMAGE PROMPTS]")
            beats: List[float] = req.audio_summary.get("beats") or req.audio_summary.get("beat_times") or []
            logger.info(f"Beats available: {len(beats)} beats")
            logger.info(f"Scene prompts available: {len(scene_prompts)} scenes")
            logger.info(f"Number of variations requested: {req.num_variations}")

            prompts: List[SeedImagePrompt] = []
            count = min(len(scene_prompts), max(1, req.num_variations))
            logger.info(f"Creating {count} seed prompts")

            for i in range(count):
                sp = scene_prompts[i]
                start_bt = float(beats[i]) if i < len(beats) else float(i * 10.0)
                end_bt = float(beats[i + 1]) if i + 1 < len(beats) else start_bt + 10.0
                txt = sp.get("detailed_prompt") or sp.get("description") or concept_final.title
                chars = sp.get("characters", [])
                ref_imgs = sp.get("reference_images", [])

                logger.info(f"\n[Seed Prompt {i+1}/{count}]")
                logger.info(f"  Time range: {start_bt:.1f}s - {end_bt:.1f}s")
                logger.info(f"  Prompt: {txt}")
                logger.info(f"  Characters: {chars}")
                logger.info(f"  Reference images: {len(ref_imgs)} images")

                prompts.append(SeedImagePrompt(
                    prompt=txt,
                    negative_prompt="low-res, blurry, duplicate",
                    start_beat=start_bt,
                    end_beat=end_bt,
                    seed=None,
                    style_tags=[concept_final.visual_style] if concept_final.visual_style else [],
                    characters=chars,
                    reference_images=ref_imgs,
                ))

            logger.info(f"\n[SEED PROMPTS CREATED] Total: {len(prompts)}")

            story = _build_story_plan(req.audio_summary, concept_final)
            resp = SeedPromptsResponse(
                concept_final=concept_final,
                story_plan=story,
                prompts=prompts,
            )
        else:
            # Fallback (no audio file): build prompts procedurally from beats and concept
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

        # If audio_file provided, use agent team to derive story first
        if req.audio_file:
            # Wait for LLM to be ready (with timeout - 10 minutes for large model downloads)
            if llm_initializing:
                logger.info("LLM is still initializing, waiting...")
                if not wait_for_llm(timeout=600):
                    return jsonify({
                        "error": "LLM initialization timed out. Please check logs and try again.",
                        "llm_initializing": True
                    }), 503
            
            if llm is None and LOCAL_LLM_AVAILABLE:
                return jsonify({
                    "error": "LLM not available. Check logs for initialization errors.",
                    "llm_error": llm_init_error
                }), 503
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                director_vision = loop.run_until_complete(
                    filmmaking_team.create_music_video_concept(
                        audio_file_path=req.audio_file,
                        user_prompt=concept_final.title,
                        lyrics=None,
                    )
                )
                scene_prompts = loop.run_until_complete(
                    filmmaking_team.generate_scene_prompts(director_vision)
                )
            finally:
                loop.close()

            # Map scenes to segments aligned to provided beat_map
            beats: List[float] = req.beat_map or req.audio_summary.get("beats") or []
            segments: List[VideoPromptSegment] = []
            count = max(len(scene_prompts), 0)
            for i in range(count):
                sp = scene_prompts[i]
                start_bt = float(beats[i]) if i < len(beats) else float(i * 10.0)
                end_bt = float(beats[i + 1]) if i + 1 < len(beats) else start_bt + 10.0
                prompt = sp.get("detailed_prompt") or sp.get("description") or concept_final.title
                segments.append(VideoPromptSegment(
                    start_beat=start_bt,
                    end_beat=end_bt,
                    prompt=prompt,
                    negative_prompt="glitches, artifacts",
                    motion_notes="motivated camera, evolving blocking",
                    transition="musical cut",
                    characters_on_screen=sp.get("characters", []),
                    reference_images=sp.get("reference_images", []),
                    scene_index=sp.get("scene_id", i),
                    beat_index=i,
                ))

            story = _build_story_plan({"beats": beats}, concept_final)
            resp = VideoPromptsResponse(
                concept_final=concept_final,
                story_plan=story,
                segments=segments,
            )
        else:
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
    port = int(os.environ.get('PORT', 5004))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting Creative Director service on port {port}")
    logger.info("Filmmaking team agents initialized and ready")

    app.run(host='0.0.0.0', port=port, debug=debug)