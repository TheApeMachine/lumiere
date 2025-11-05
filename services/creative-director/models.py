from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class Character(BaseModel):
    id: str
    name: Optional[str] = None
    role: Optional[str] = None
    descriptors: List[str] = []
    reference_images: List[str] = []
    identity_token: Optional[str] = None


class Concept(BaseModel):
    title: str
    theme: Optional[str] = None
    mood: Optional[str] = None
    visual_style: Optional[str] = None
    characters: List[Character] = []


class StoryBeat(BaseModel):
    beat_index: int
    label: str
    start_beat: float
    end_beat: float


class Scene(BaseModel):
    scene_index: int
    location: str
    time_of_day: str
    characters_on_stage: List[str]
    palette: List[str] = []
    beats: List[StoryBeat]


class StoryPlan(BaseModel):
    logline: str
    acts: int = 3
    scenes: List[Scene]


class SeedImagePrompt(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    start_beat: float
    end_beat: float
    seed: Optional[int] = None
    style_tags: List[str] = []
    characters: List[str] = []
    reference_images: List[str] = []


class VideoPromptSegment(BaseModel):
    start_beat: float
    end_beat: float
    prompt: str
    negative_prompt: Optional[str] = None
    motion_notes: Optional[str] = None
    transition: Optional[str] = None
    characters_on_screen: List[str] = []
    reference_images: List[str] = []
    scene_index: int
    beat_index: int


class SeedPromptsRequest(BaseModel):
    project_id: str
    audio_summary: Dict[str, Any]
    concept: Concept
    num_variations: int = 6


class SeedPromptsResponse(BaseModel):
    concept_final: Concept
    story_plan: StoryPlan
    prompts: List[SeedImagePrompt]


class VideoPromptsRequest(BaseModel):
    project_id: str
    audio_summary: Dict[str, Any]
    concept: Concept
    beat_map: List[float]


class VideoPromptsResponse(BaseModel):
    concept_final: Concept
    story_plan: StoryPlan
    segments: List[VideoPromptSegment]


