from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import io
import json
import tempfile
import torchaudio
from pyannote.audio import Pipeline

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Faster Whisper model
model_path = "./largev3"  # Adjust to your model path
model = WhisperModel(model_path, device="cuda", compute_type="int8_float16")

# Initialize PyAnnote pipeline (Load once)
hf_token = "enter your huggingface token"  # <-- Replace with your actual HF token
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the CAIR Transcriber"}

def is_overlap(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)

def stream_transcription(audio_data: bytes, target_language: str = None):
    try:
        # Save to temp file for diarization
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        yield json.dumps({
            "type": "info",
            "data": "Model loaded"
        })

        # Transcribe using Faster Whisper
        task = "transcribe"
        segments, info = model.transcribe(io.BytesIO(audio_data), beam_size=5, task=task)

        yield json.dumps({
            "type": "language_detection",
            "data": info.language
        })

        # Run speaker diarization using pyannote
        diarization = diarization_pipeline(tmp_path)

# Build a list of speaker turns
        speaker_turns = list(diarization.itertracks(yield_label=True))

        # Map real speaker labels to Speaker 1, 2, 3...
        speaker_id_map = {}
        speaker_counter = 1

        transcript_with_speakers = []
        for segment in segments:
            speaker_overlap_counts = {}
            for turn, _, real_speaker in speaker_turns:
                if is_overlap(segment.start, segment.end, turn.start, turn.end):
                    speaker_overlap_counts[real_speaker] = speaker_overlap_counts.get(real_speaker, 0) + 1

            if speaker_overlap_counts:
                # Get the speaker with the most overlaps
                matched_speaker = max(speaker_overlap_counts, key=speaker_overlap_counts.get)
                if matched_speaker not in speaker_id_map:
                    speaker_id_map[matched_speaker] = f"Speaker {speaker_counter}"
                    speaker_counter += 1
                speaker_label = speaker_id_map[matched_speaker]
            else:
                speaker_label = f"Speaker {speaker_counter}"
                speaker_counter += 1

            transcript_with_speakers.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker_label,
                "text": segment.text.strip()
            })

        # Yield the final segments with speaker tags
        for item in transcript_with_speakers:
            yield json.dumps(item)

    except Exception as e:
        yield json.dumps({
            "type": "error",
            "data": f"Error during transcription: {str(e)}"
        })
    finally:
        if 'tmp_path' in locals() and tmp_path:
            import os
            os.remove(tmp_path)

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    target_language: str = Form(None)
):
    try:
        audio_data = await file.read()
        if len(audio_data) > 100_000_000:
            return JSONResponse(content={
                "type": "error",
                "data": "File size exceeds 100MB limit"
            }, status_code=400)

        return StreamingResponse(
            stream_transcription(audio_data, target_language),
            media_type="application/json"
        )

    except Exception as e:
        return JSONResponse(content={
            "type": "error",
            "data": f"Error during file upload: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
