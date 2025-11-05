import json
from server import app


def test_seed_prompts_endpoint():
    client = app.test_client()
    payload = {
        "project_id": "test_proj",
        "audio_summary": {"bpm": 120, "beats": [0, 1, 2, 3]},
        "concept": {"title": "Test Title", "characters": []},
        "num_variations": 3,
    }
    resp = client.post("/v1/seed_prompts", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "prompts" in data and len(data["prompts"]) >= 1
    assert "concept_final" in data


