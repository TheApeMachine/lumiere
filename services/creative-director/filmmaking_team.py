"""
Filmmaking Team - Agent-based creative system for dynamic music video generation.

This module implements a multi-agent system where specialized agents collaborate
to create compelling narratives and visuals for music videos.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Use local agents instead of OpenAI Agents SDK
try:
    from local_agents import Agent, Runner, function_tool, handoff, LocalLLM, AgentContext, RECOMMENDED_PROMPT_PREFIX
    LOCAL_AGENTS_AVAILABLE = True
except ImportError:
    # Fallback to OpenAI if local agents not available
    try:
        from agents import Agent, Runner, function_tool, handoff
        from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
        LOCAL_AGENTS_AVAILABLE = False
    except ImportError:
        raise ImportError("Neither local_agents nor openai-agents available. Install llama-cpp-python for local mode.")


# Data Models
class MusicAnalysis(BaseModel):
    """Analysis of the input music track."""
    genre: str
    tempo: str  # slow, medium, fast
    mood: str
    energy_level: str  # low, medium, high
    instruments: List[str]
    lyrics_themes: List[str]
    key_moments: List[Dict[str, Any]]  # timestamp, description, intensity


class StoryTreatment(BaseModel):
    """Story treatment from the writer agent."""
    title: str
    logline: str
    narrative_arc: str
    character_descriptions: List[str]
    key_scenes: List[Dict[str, Any]]
    visual_themes: List[str]
    tone: str


class CinematographyPlan(BaseModel):
    """Cinematography plan from the DoP agent."""
    visual_style: str
    color_palette: List[str]
    camera_movements: List[str]
    lighting_style: str
    shot_compositions: List[Dict[str, Any]]
    transitions: List[str]


class DirectorVision(BaseModel):
    """Final creative vision from the director."""
    approved_story: StoryTreatment
    approved_cinematography: CinematographyPlan
    scene_breakdown: List[Dict[str, Any]]
    revision_notes: Optional[str]
    final_concept: str


@dataclass
class FilmmakingContext:
    """Context shared across all agents in the filmmaking team."""
    music_analysis: Optional[MusicAnalysis] = None
    story_treatment: Optional[StoryTreatment] = None
    cinematography_plan: Optional[CinematographyPlan] = None
    director_vision: Optional[DirectorVision] = None
    revision_count: int = 0
    max_revisions: int = 3


class FilmmakingTeam:
    """
    Multi-agent filmmaking team that creates dynamic, story-driven music videos.

    The team consists of:
    - Music Analyst: Analyzes the audio track
    - Writer: Creates narrative treatments
    - Director of Photography (DoP): Plans cinematography
    - Director: Makes final creative decisions and requests revisions
    """

    def __init__(self, audio_intelligence=None, llm: Optional[LocalLLM] = None):
        self.logger = logging.getLogger(__name__)
        self.audio_intelligence = audio_intelligence
        self.llm = llm
        
        # Warn if no LLM is provided
        if self.llm is None:
            self.logger.warning("No LLM provided to FilmmakingTeam. Agents will not be able to generate responses.")
            self.logger.warning("Please ensure LOCAL_LLM_MODEL_PATH or LOCAL_LLM_MODEL is set, or OPENAI_API_KEY is configured.")
        
        self._setup_agents()

    def _setup_agents(self):
        """Initialize all agents in the filmmaking team."""

        # Music Analyst Agent
        self.music_analyst = Agent(
            name="Music Analyst",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

            You are an expert music analyst specializing in music video production.
            Your role is to deeply analyze audio tracks to extract creative insights.

            Analyze the music for:
            - Genre and subgenre classification
            - Tempo and rhythm patterns
            - Emotional mood and energy levels
            - Instrumental composition
            - Lyrical themes (if lyrics provided)
            - Key musical moments that could drive visual storytelling

            Provide detailed, actionable insights that will inform the creative team.
            Be specific about timestamps for key moments and transitions.

            Output your analysis as valid JSON matching this structure:
            {{
                "genre": "string",
                "tempo": "slow|medium|fast",
                "mood": "string",
                "energy_level": "low|medium|high",
                "instruments": ["string"],
                "lyrics_themes": ["string"],
                "key_moments": [{{"timestamp": float, "description": "string", "intensity": float}}]
            }}
            """,
            tools=[self._analyze_music_tool],
            llm=self.llm
        )

        # Writer Agent
        self.writer = Agent(
            name="Writer",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

            You are a creative writer specializing in music video narratives.
            Your role is to create compelling story treatments based on music analysis.

            Create narratives that:
            - Complement the music's emotional journey
            - Include interesting characters and conflicts
            - Have clear visual storytelling potential
            - Match the genre and mood of the music
            - Utilize the key musical moments for dramatic beats

            Think cinematically - every scene should be visually striking.
            Avoid clichés and create fresh, engaging concepts.

            Output your story treatment as valid JSON matching this structure:
            {{
                "title": "string",
                "logline": "string",
                "narrative_arc": "string",
                "character_descriptions": ["string"],
                "key_scenes": [{{"timestamp": float, "description": "string", "mood": "string"}}],
                "visual_themes": ["string"],
                "tone": "string"
            }}
            """,
            tools=[self._create_story_treatment_tool],
            handoffs=[handoff(self.music_analyst, tool_name_override="request_music_reanalysis")],
            llm=self.llm
        )

        # Director of Photography Agent
        self.dop = Agent(
            name="Director of Photography",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

            You are an expert cinematographer specializing in music videos.
            Your role is to create visual plans that bring stories to life.

            Design cinematography that:
            - Enhances the narrative and emotional impact
            - Matches the music's energy and rhythm
            - Uses innovative camera work and lighting
            - Creates a cohesive visual style
            - Plans specific shots for key musical moments

            Consider color psychology, movement, and composition.
            Think about how visuals can dance with the music.

            Output your cinematography plan as valid JSON matching this structure:
            {{
                "visual_style": "string",
                "color_palette": ["string"],
                "camera_movements": ["string"],
                "lighting_style": "string",
                "shot_compositions": [{{"type": "string", "description": "string"}}],
                "transitions": ["string"]
            }}
            """,
            tools=[self._create_cinematography_plan_tool],
            handoffs=[
                handoff(self.writer, tool_name_override="request_story_revision"),
                handoff(self.music_analyst, tool_name_override="request_music_reanalysis")
            ],
            llm=self.llm
        )

        # Director Agent (Main orchestrator)
        self.director = Agent(
            name="Director",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

            You are the creative director of this music video production.
            Your role is to review all creative elements and make final decisions.

            Evaluate proposals for:
            - Creative originality and impact
            - Coherence between story, visuals, and music
            - Production feasibility
            - Overall artistic vision

            You can approve concepts or request revisions with specific feedback.
            Be demanding but constructive - push for excellence.
            Consider the target audience and commercial viability.

            Output your final vision as valid JSON matching this structure:
            {{
                "approved_story": {{...story treatment...}},
                "approved_cinematography": {{...cinematography plan...}},
                "scene_breakdown": [{{"timestamp": float, "duration": float, "description": "string", "mood": "string", "characters": ["string"], "setting": "string", "camera_movement": "string"}}],
                "revision_notes": "string or null",
                "final_concept": "string"
            }}
            """,
            tools=[
                self._approve_concept_tool,
                self._request_revision_tool
            ],
            handoffs=[
                handoff(self.writer, tool_name_override="request_story_revision"),
                handoff(self.dop, tool_name_override="request_cinematography_revision"),
                handoff(self.music_analyst, tool_name_override="request_music_reanalysis")
            ],
            llm=self.llm
        )

    @staticmethod
    @function_tool
    def _analyze_music_tool(audio_file_path: str, lyrics: Optional[str] = None) -> str:
        """Analyze music track for creative insights."""
        # This would integrate with actual audio analysis
        # For now, return a placeholder that triggers proper analysis
        return f"Analyzing audio file: {audio_file_path}"

    @staticmethod
    @function_tool
    def _create_story_treatment_tool(music_analysis: str, creative_brief: str) -> str:
        """Create a story treatment based on music analysis."""
        return f"Creating story treatment based on: {music_analysis}"

    @staticmethod
    @function_tool
    def _create_cinematography_plan_tool(story_treatment: str, music_analysis: str) -> str:
        """Create cinematography plan based on story and music."""
        return f"Creating cinematography plan for: {story_treatment}"

    @staticmethod
    @function_tool
    def _approve_concept_tool(concept_summary: str) -> str:
        """Approve the final concept."""
        return f"Approving concept: {concept_summary}"

    @staticmethod
    @function_tool
    def _request_revision_tool(revision_notes: str, target_agent: str) -> str:
        """Request revisions from specific agent."""
        return f"Requesting revision from {target_agent}: {revision_notes}"

    async def create_music_video_concept(
        self,
        audio_file_path: str,
        user_prompt: str = "",
        lyrics: Optional[str] = None
    ) -> DirectorVision:
        """
        Main method to create a music video concept using the agent team.

        Args:
            audio_file_path: Path to the audio file
            user_prompt: Optional creative direction from user
            lyrics: Optional lyrics text

        Returns:
            DirectorVision: Final approved creative concept
        """
        # Use AgentContext for local agents, FilmmakingContext for OpenAI agents
        if LOCAL_AGENTS_AVAILABLE:
            context = AgentContext()
        else:
            context = FilmmakingContext()

        try:
            # Step 1: Music Analysis - Use raw audio data, let LLM interpret creatively
            self.logger.info("=" * 80)
            self.logger.info("FILMMAKING PIPELINE: Starting music video concept creation")
            self.logger.info("=" * 80)
            self.logger.info(f"Input - audio_file_path: {audio_file_path}")
            self.logger.info(f"Input - user_prompt: {user_prompt}")
            self.logger.info(f"Input - lyrics provided: {lyrics is not None}")

            self.logger.info("\n[STEP 1/4] Starting music analysis...")
            raw_audio_data = None
            if self.audio_intelligence:
                # Get RAW audio features - no heuristics, no hardcoded interpretations
                try:
                    self.logger.info(f"Calling audio_intelligence.analyze_audio_file('{audio_file_path}')")
                    audio_analysis = self.audio_intelligence.analyze_audio_file(audio_file_path)
                    self.logger.info("✓ Extracted raw audio features from librosa")
                    self.logger.info(f"Raw audio analysis structure keys: {list(audio_analysis.keys())}")

                    # Pass RAW data - let the LLM interpret creatively
                    musical_analysis = audio_analysis.get('musical_analysis', {})
                    tempo_data = musical_analysis.get('tempo', {})
                    file_info = audio_analysis.get('file_info', {})
                    song_structure = audio_analysis.get('song_structure', {})
                    key_moments_raw = audio_analysis.get('key_moments', [])

                    self.logger.info(f"Tempo data: {json.dumps(tempo_data, indent=2)}")
                    self.logger.info(f"File info: duration={file_info.get('duration', 0):.1f}s, sr={file_info.get('sample_rate', 0)}")
                    self.logger.info(f"Song structure: {len(song_structure.get('segments', []))} segments")
                    self.logger.info(f"Key moments: {len(key_moments_raw)} moments detected")

                    # Extract raw numerical data
                    raw_audio_data = {
                        'duration': file_info.get('duration', 0),
                        'sample_rate': file_info.get('sample_rate', 0),
                        'tempo_bpm': tempo_data.get('bpm', 0),
                        'tempo_variance': tempo_data.get('variance', 0),
                        'key': musical_analysis.get('key', 'unknown'),
                        'mode': musical_analysis.get('mode', 'unknown'),
                        'time_signature': musical_analysis.get('time_signature', 'unknown'),
                        'energy_level': audio_analysis.get('mood_and_energy', {}).get('energy_level', 'unknown'),
                        'energy_variance': audio_analysis.get('mood_and_energy', {}).get('energy_variance', 0),
                        'spectral_brightness': audio_analysis.get('mood_and_energy', {}).get('spectral_brightness', 0),
                        'instruments_detected': [
                            {
                                'name': inst.get('instrument', ''),
                                'confidence': inst.get('confidence', 0),
                                'prominence': inst.get('prominence', 'unknown')
                            }
                            for inst in audio_analysis.get('instruments', [])
                        ],
                        'song_segments': [
                            {
                                'type': seg.get('type', 'unknown'),
                                'start_time': seg.get('start_time', 0),
                                'end_time': seg.get('end_time', 0),
                                'duration': seg.get('duration', 0)
                            }
                            for seg in song_structure.get('segments', [])
                        ],
                        'key_moments': [
                            {
                                'timestamp': float(m.get('timestamp', 0.0)),
                                'type': m.get('type', 'unknown'),
                                'intensity': float(m.get('intensity', 0.5))
                            }
                            for m in key_moments_raw
                        ]
                    }
                    self.logger.info("\n[RAW AUDIO DATA EXTRACTED]")
                    self.logger.info(json.dumps(raw_audio_data, indent=2, default=str))
                except Exception as e:
                    self.logger.error(f"✗ Error extracting raw audio data: {e}", exc_info=True)
                    raw_audio_data = None

            # Ask LLM to analyze with raw data - NO hardcoded interpretations
            if raw_audio_data:
                self.logger.info("\n[PREPARING LLM PROMPT FOR MUSIC ANALYST]")
                music_input = f"""Analyze this audio file and provide creative insights for music video production.

RAW AUDIO DATA (interpret this creatively, don't use preset templates):
- Duration: {raw_audio_data['duration']:.1f} seconds
- Tempo: {raw_audio_data['tempo_bpm']:.1f} BPM (variance: {raw_audio_data['tempo_variance']:.2f})
- Musical Key: {raw_audio_data['key']} {raw_audio_data['mode']}
- Time Signature: {raw_audio_data['time_signature']}
- Energy Level: {raw_audio_data['energy_level']} (variance: {raw_audio_data['energy_variance']:.4f})
- Spectral Brightness: {raw_audio_data['spectral_brightness']:.1f} Hz
- Detected Instruments: {json.dumps(raw_audio_data['instruments_detected'], indent=2)}
- Song Structure: {len(raw_audio_data['song_segments'])} segments - {json.dumps(raw_audio_data['song_segments'], indent=2)}
- Key Musical Moments: {len(raw_audio_data['key_moments'])} moments at {', '.join([f"{m['timestamp']:.1f}s" for m in raw_audio_data['key_moments'][:10]])}

Analyze these raw audio characteristics and create ORIGINAL, CREATIVE insights for visual storytelling.
Do NOT use generic templates based on genre. Interpret the actual musical data creatively and uniquely.
Think about what makes this specific track interesting and build a unique visual concept around it."""
            else:
                music_input = f"Analyze this audio file: {audio_file_path}"

            if lyrics:
                music_input += f"\n\nLYRICS:\n{lyrics}\n\nIncorporate lyrical themes and narrative elements into your analysis."
                self.logger.info(f"Lyrics length: {len(lyrics)} characters")
            if user_prompt:
                music_input += f"\n\nUser Creative Direction: {user_prompt}"
                self.logger.info(f"User prompt: {user_prompt}")

            self.logger.info("\n[MUSIC ANALYST AGENT INPUT]")
            self.logger.info(f"{'=' * 80}")
            self.logger.info(music_input)
            self.logger.info(f"{'=' * 80}")

            self.logger.info("\n[RUNNING MUSIC ANALYST AGENT...]")
            music_result = await Runner.run(self.music_analyst, music_input, context=context)
            self.logger.info("✓ Music Analyst agent completed")
            self.logger.info(f"Raw output type: {type(music_result.final_output)}")
            self.logger.info(f"Raw output (first 500 chars): {str(music_result.final_output)[:500]}")

            # Parse LLM's creative interpretation - let it decide everything
            self.logger.info("\n[PARSING MUSIC ANALYST OUTPUT]")
            parsed_analysis = self._parse_output(music_result.final_output, MusicAnalysis)
            context.music_analysis = parsed_analysis

            self.logger.info(f"Parsed analysis type: {type(parsed_analysis)}")
            if isinstance(parsed_analysis, MusicAnalysis):
                self.logger.info("\n[PARSED MUSIC ANALYSIS]")
                self.logger.info(f"  Genre: {parsed_analysis.genre}")
                self.logger.info(f"  Tempo: {parsed_analysis.tempo}")
                self.logger.info(f"  Mood: {parsed_analysis.mood}")
                self.logger.info(f"  Energy Level: {parsed_analysis.energy_level}")
                self.logger.info(f"  Instruments: {parsed_analysis.instruments}")
                self.logger.info(f"  Lyrics Themes: {parsed_analysis.lyrics_themes}")
                self.logger.info(f"  Key Moments: {len(parsed_analysis.key_moments)} moments")
                for i, km in enumerate(parsed_analysis.key_moments[:5]):
                    self.logger.info(f"    {i+1}. {km.get('timestamp', 0):.1f}s - {km.get('description', 'N/A')}")

            # If parsing failed, create a minimal structure from the raw output
            if not isinstance(context.music_analysis, MusicAnalysis):
                # Extract what we can from the raw text
                output_text = str(music_result.final_output)
                context.music_analysis = MusicAnalysis(
                    genre="interpreted from audio",
                    tempo="interpreted from audio",
                    mood="interpreted from audio",
                    energy_level="interpreted from audio",
                    instruments=[],
                    lyrics_themes=[],
                    key_moments=raw_audio_data.get('key_moments', []) if raw_audio_data else []
                )

            # Step 2: Story Treatment
            self.logger.info("\n" + "=" * 80)
            self.logger.info("[STEP 2/4] Creating story treatment...")
            story_input = f"Create a compelling narrative based on this music analysis: {context.music_analysis}"
            if user_prompt:
                story_input += f"\nUser creative direction: {user_prompt}"

            self.logger.info("\n[WRITER AGENT INPUT]")
            self.logger.info(f"{'=' * 80}")
            self.logger.info(story_input)
            self.logger.info(f"{'=' * 80}")

            self.logger.info("\n[RUNNING WRITER AGENT...]")
            story_result = await Runner.run(
                self.writer,
                story_input,
                context=context
            )
            self.logger.info("✓ Writer agent completed")
            self.logger.info(f"Raw output (first 500 chars): {str(story_result.final_output)[:500]}")

            # Parse output to StoryTreatment model
            self.logger.info("\n[PARSING WRITER OUTPUT]")
            context.story_treatment = self._parse_output(story_result.final_output, StoryTreatment)

            if isinstance(context.story_treatment, StoryTreatment):
                self.logger.info("\n[PARSED STORY TREATMENT]")
                self.logger.info(f"  Title: {context.story_treatment.title}")
                self.logger.info(f"  Logline: {context.story_treatment.logline}")
                self.logger.info(f"  Narrative Arc: {context.story_treatment.narrative_arc}")
                self.logger.info(f"  Tone: {context.story_treatment.tone}")
                self.logger.info(f"  Characters: {len(context.story_treatment.character_descriptions)} characters")
                self.logger.info(f"  Key Scenes: {len(context.story_treatment.key_scenes)} scenes")
                self.logger.info(f"  Visual Themes: {context.story_treatment.visual_themes}")

            # Step 3: Cinematography Planning
            self.logger.info("\n" + "=" * 80)
            self.logger.info("[STEP 3/4] Planning cinematography...")
            dop_input = f"""Plan the cinematography for this project:
            Music Analysis: {context.music_analysis}
            Story Treatment: {context.story_treatment}
            """

            self.logger.info("\n[DOP AGENT INPUT]")
            self.logger.info(f"{'=' * 80}")
            self.logger.info(dop_input)
            self.logger.info(f"{'=' * 80}")

            self.logger.info("\n[RUNNING DOP AGENT...]")
            dop_result = await Runner.run(
                self.dop,
                dop_input,
                context=context
            )
            self.logger.info("✓ DOP agent completed")
            self.logger.info(f"Raw output (first 500 chars): {str(dop_result.final_output)[:500]}")

            # Parse output to CinematographyPlan model
            self.logger.info("\n[PARSING DOP OUTPUT]")
            context.cinematography_plan = self._parse_output(dop_result.final_output, CinematographyPlan)

            if isinstance(context.cinematography_plan, CinematographyPlan):
                self.logger.info("\n[PARSED CINEMATOGRAPHY PLAN]")
                self.logger.info(f"  Visual Style: {context.cinematography_plan.visual_style}")
                self.logger.info(f"  Color Palette: {context.cinematography_plan.color_palette}")
                self.logger.info(f"  Lighting Style: {context.cinematography_plan.lighting_style}")
                self.logger.info(f"  Camera Movements: {context.cinematography_plan.camera_movements}")
                self.logger.info(f"  Transitions: {context.cinematography_plan.transitions}")
                self.logger.info(f"  Shot Compositions: {len(context.cinematography_plan.shot_compositions)} shots")

            # Step 4: Director Review and Approval
            self.logger.info("\n" + "=" * 80)
            self.logger.info("[STEP 4/4] Director reviewing concept...")
            director_input = f"""Review this complete music video concept:

            Music Analysis: {context.music_analysis}
            Story Treatment: {context.story_treatment}
            Cinematography Plan: {context.cinematography_plan}

            Evaluate for creativity, coherence, and production value.
            Approve or request specific revisions.
            """

            self.logger.info("\n[DIRECTOR AGENT INPUT]")
            self.logger.info(f"{'=' * 80}")
            self.logger.info(director_input)
            self.logger.info(f"{'=' * 80}")

            # Revision loop
            while context.revision_count < context.max_revisions:
                self.logger.info(f"\n[RUNNING DIRECTOR AGENT (attempt {context.revision_count + 1}/{context.max_revisions})...]")
                director_result = await Runner.run(
                    self.director,
                    director_input,
                    context=context
                )
                self.logger.info("✓ Director agent completed")
                self.logger.info(f"Raw output (first 500 chars): {str(director_result.final_output)[:500]}")

                # Parse output to DirectorVision model
                self.logger.info("\n[PARSING DIRECTOR OUTPUT]")
                parsed_vision = self._parse_output(director_result.final_output, DirectorVision)

                if isinstance(parsed_vision, DirectorVision):
                    self.logger.info("\n[PARSED DIRECTOR VISION]")
                    self.logger.info(f"  Final Concept: {parsed_vision.final_concept}")
                    self.logger.info(f"  Revision Notes: {parsed_vision.revision_notes}")
                    self.logger.info(f"  Scene Breakdown: {len(parsed_vision.scene_breakdown)} scenes")
                    if parsed_vision.approved_story:
                        self.logger.info(f"  Approved Story Title: {parsed_vision.approved_story.title}")
                    if parsed_vision.approved_cinematography:
                        self.logger.info(f"  Approved Visual Style: {parsed_vision.approved_cinematography.visual_style}")

                # Check if director approved or wants revisions
                is_approved = self._is_concept_approved(parsed_vision)
                self.logger.info(f"Concept approved: {is_approved}")

                if is_approved:
                    context.director_vision = parsed_vision
                    self.logger.info("✓ Director approved the concept!")
                    break
                else:
                    # Handle revision request
                    context.revision_count += 1
                    self.logger.info(f"⚠ Revision requested (attempt {context.revision_count}/{context.max_revisions})")
                    # In a full implementation, this would route back to specific agents

            if not context.director_vision:
                # Max revisions reached, use last result
                self.logger.info("⚠ Max revisions reached, using last result")
                context.director_vision = self._parse_output(director_result.final_output, DirectorVision)

            self.logger.info("\n" + "=" * 80)
            self.logger.info("✓ Music video concept creation completed!")
            self.logger.info("=" * 80)
            return context.director_vision

        except Exception as e:
            self.logger.error(f"Error in concept creation: {e}")
            raise

    def _parse_output(self, output: Any, model_class: type) -> Any:
        """Parse agent output to the specified Pydantic model."""
        self.logger.info(f"[PARSE] Attempting to parse as {model_class.__name__}")
        self.logger.info(f"[PARSE] Input type: {type(output)}")

        if isinstance(output, model_class):
            self.logger.info(f"[PARSE] Already a {model_class.__name__} instance")
            return output
        if isinstance(output, dict):
            self.logger.info(f"[PARSE] Input is dict, converting to {model_class.__name__}")
            self.logger.info(f"[PARSE] Dict keys: {list(output.keys())}")
            try:
                result = model_class(**output)
                self.logger.info(f"[PARSE] ✓ Successfully converted dict to {model_class.__name__}")
                return result
            except Exception as e:
                self.logger.warning(f"[PARSE] ✗ Failed to parse dict as {model_class.__name__}: {e}")
                return output
        if isinstance(output, str):
            self.logger.info(f"[PARSE] Input is string, length: {len(output)}")
            self.logger.info(f"[PARSE] String preview: {output[:200]}...")

            # Extract JSON from markdown code blocks if present
            json_str = output.strip()
            if json_str.startswith("```"):
                self.logger.info("[PARSE] Detected markdown code block, extracting JSON")
                # Extract content between ```json and ```
                match = re.search(r'```(?:json)?\s*\n(.*?)\n```', json_str, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    self.logger.info(f"[PARSE] Extracted JSON from markdown block")
                else:
                    # Try without language identifier
                    json_str = re.sub(r'^```\s*\n', '', json_str)
                    json_str = re.sub(r'\n```\s*$', '', json_str)

            # Extract the first complete JSON object (handles extra data after JSON)
            # This is more precise than regex - we count braces to find the exact end
            self.logger.info("[PARSE] Extracting first complete JSON object...")
            brace_count = 0
            start_idx = json_str.find('{')
            if start_idx >= 0:
                # Also need to handle string literals which might contain braces
                in_string = False
                escape_next = False

                for i in range(start_idx, len(json_str)):
                    char = json_str[i]

                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found the complete JSON object
                                json_obj_str = json_str[start_idx:i+1]
                                self.logger.info(f"[PARSE] Extracted complete JSON object ({len(json_obj_str)} chars)")
                                json_str = json_obj_str
                                break
            else:
                self.logger.warning("[PARSE] No opening brace found in string")

            try:
                self.logger.info(f"[PARSE] Attempting JSON parse...")
                data = json.loads(json_str)
                self.logger.info(f"[PARSE] ✓ JSON parsed successfully, keys: {list(data.keys())}")
                result = model_class(**data)
                self.logger.info(f"[PARSE] ✓ Successfully created {model_class.__name__} instance")
                return result
            except json.JSONDecodeError as e:
                self.logger.warning(f"[PARSE] ✗ JSON decode failed: {e}")
                self.logger.info(f"[PARSE] Failed JSON string (first 500 chars): {json_str[:500]}")
                self.logger.info(f"[PARSE] Failed JSON string (last 200 chars): {json_str[-200:]}")
                # Return the raw string if parsing fails - caller should handle this
                self.logger.warning(f"[PARSE] Returning raw output (could not parse as {model_class.__name__})")
                return output
        return output

    def _is_concept_approved(self, director_output: Any) -> bool:
        """Check if the director approved the concept."""
        # This would check the director's output for approval indicators
        # For now, assume approval if no explicit revision request
        if isinstance(director_output, str):
            return "revision" not in director_output.lower()
        if isinstance(director_output, DirectorVision):
            # Check revision_notes field
            if director_output.revision_notes:
                return "revision" not in director_output.revision_notes.lower()
        return True

    async def generate_scene_prompts(self, director_vision: DirectorVision) -> List[Dict[str, Any]]:
        """
        Generate detailed prompts for each scene based on the director's vision.

        Returns:
            List of scene prompts with timestamps, descriptions, and visual details
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[GENERATING SCENE PROMPTS]")
        self.logger.info(f"Director vision type: {type(director_vision)}")
        self.logger.info(f"Scene breakdown length: {len(director_vision.scene_breakdown) if hasattr(director_vision, 'scene_breakdown') else 0}")

        scene_prompts = []

        for i, scene in enumerate(director_vision.scene_breakdown):
            self.logger.info(f"\n[Processing Scene {i+1}/{len(director_vision.scene_breakdown)}]")
            self.logger.info(f"Scene data: {json.dumps(scene if isinstance(scene, dict) else scene.__dict__ if hasattr(scene, '__dict__') else str(scene), indent=2, default=str)}")
            detailed_prompt = self._create_detailed_scene_prompt(scene, director_vision)

            prompt = {
                "scene_id": i,
                "timestamp": scene.get("timestamp", 0),
                "duration": scene.get("duration", 3.0),
                "description": scene.get("description", ""),
                "visual_style": director_vision.approved_cinematography.visual_style if hasattr(director_vision, 'approved_cinematography') and director_vision.approved_cinematography else "cinematic",
                "color_palette": director_vision.approved_cinematography.color_palette if hasattr(director_vision, 'approved_cinematography') and director_vision.approved_cinematography else [],
                "camera_movement": scene.get("camera_movement", "static"),
                "lighting": director_vision.approved_cinematography.lighting_style if hasattr(director_vision, 'approved_cinematography') and director_vision.approved_cinematography else "natural",
                "characters": scene.get("characters", []),
                "setting": scene.get("setting", ""),
                "mood": scene.get("mood", "neutral"),
                "detailed_prompt": detailed_prompt
            }
            scene_prompts.append(prompt)

            self.logger.info(f"  Scene {i+1} prompt created:")
            self.logger.info(f"    Timestamp: {prompt['timestamp']}s")
            self.logger.info(f"    Description: {prompt['description'][:100]}...")
            self.logger.info(f"    Detailed Prompt: {detailed_prompt[:200]}...")

        self.logger.info(f"\n[SCENE PROMPTS GENERATED]")
        self.logger.info(f"Total scenes: {len(scene_prompts)}")
        self.logger.info("=" * 80)
        return scene_prompts

    def _create_detailed_scene_prompt(self, scene: Dict[str, Any], vision: DirectorVision) -> str:
        """Create a detailed prompt for image/video generation, respecting token limits."""
        self.logger.info(f"[PROMPT] Creating detailed prompt for scene")
        self.logger.info(f"[PROMPT] Scene data: {json.dumps(scene if isinstance(scene, dict) else scene.__dict__ if hasattr(scene, '__dict__') else str(scene), indent=2, default=str)}")

        # Build rich, detailed prompt from director's vision, prioritizing key elements
        parts = []

        # Core description is highest priority
        description = scene.get('description', '')
        if description:
            parts.append(description)

        # Add visual style and key aesthetic elements
        if hasattr(vision, 'approved_cinematography') and vision.approved_cinematography:
            approved_cinematography = vision.approved_cinematography
            if hasattr(approved_cinematography, 'visual_style') and approved_cinematography.visual_style:
                # Summarize visual style if too long
                style = approved_cinematography.visual_style
                if len(style.split()) > 15:
                    style = f"style of {', '.join(style.split()[:10])}"
                parts.append(style)

            if hasattr(approved_cinematography, 'color_palette') and approved_cinematography.color_palette:
                parts.append(f"color palette of {', '.join(approved_cinematography.color_palette[:3])}")

            if hasattr(approved_cinematography, 'lighting_style') and approved_cinematography.lighting_style:
                parts.append(f"lighting: {approved_cinematography.lighting_style.split('.')[0]}")


        # Add other specific details if there's room
        setting = scene.get('setting', '')
        if setting:
            parts.append(f"setting is {setting}")

        characters = scene.get('characters', [])
        if characters:
            parts.append(f"featuring {' and '.join(characters)}")

        camera = scene.get('camera_movement', '')
        if camera:
            parts.append(f"camera: {camera.split(',')[0]}")

        mood = scene.get('mood', '')
        if mood:
            parts.append(f"mood is {mood}")

        # Final quality tags
        parts.append("cinematic, professional music video, high detail")

        # Join and truncate to a safe length for CLIP (approx. 70 tokens)
        final_prompt = ", ".join(parts)
        if len(final_prompt.split()) > 70:
            final_prompt = ", ".join(final_prompt.split()[:70])


        self.logger.info(f"[PROMPT] Final detailed prompt length: {len(final_prompt.split())} words")
        self.logger.info(f"[PROMPT] Final prompt: {final_prompt}")
        return final_prompt


# Audio Analysis Integration
class AudioAnalyzer:
    """Integrates with audio analysis tools to extract musical features."""

    @staticmethod
    async def analyze_audio_file(file_path: str) -> MusicAnalysis:
        """
        Analyze audio file and return structured music data.
        This would integrate with actual audio analysis libraries.
        """
        # Placeholder implementation
        # In reality, this would use librosa, essentia, or similar
        return MusicAnalysis(
            genre="electronic",
            tempo="medium",
            mood="energetic",
            energy_level="high",
            instruments=["synthesizer", "drums", "bass"],
            lyrics_themes=["freedom", "adventure"],
            key_moments=[
                {"timestamp": 0.0, "description": "intro buildup", "intensity": 0.3},
                {"timestamp": 30.0, "description": "first drop", "intensity": 0.9},
                {"timestamp": 60.0, "description": "breakdown", "intensity": 0.4},
                {"timestamp": 90.0, "description": "final climax", "intensity": 1.0}
            ]
        )


# Lyrics Processing
class LyricsProcessor:
    """Processes lyrics to extract themes and narrative elements."""

    @staticmethod
    def extract_themes(lyrics: str) -> List[str]:
        """Extract thematic elements from lyrics."""
        # This would use NLP to analyze lyrics
        # Placeholder implementation
        common_themes = ["love", "freedom", "struggle", "celebration", "journey", "dreams"]
        return common_themes[:3]  # Return first 3 as example

    @staticmethod
    def identify_narrative_structure(lyrics: str) -> Dict[str, Any]:
        """Identify narrative elements in lyrics."""
        return {
            "has_story": True,
            "characters": ["protagonist"],
            "setting": "urban",
            "conflict": "internal struggle",
            "resolution": "triumph"
        }


if __name__ == "__main__":
    # Example usage
    async def main():
        team = FilmmakingTeam()

        # Example concept creation
        vision = await team.create_music_video_concept(
            audio_file_path="/path/to/audio.mp3",
            user_prompt="Create something futuristic and energetic",
            lyrics="Sample lyrics about freedom and adventure"
        )

        print("Director's Vision:")
        print(json.dumps(vision.dict(), indent=2))

        # Generate scene prompts
        scene_prompts = await team.generate_scene_prompts(vision)
        print(f"\nGenerated {len(scene_prompts)} scene prompts")

    asyncio.run(main())