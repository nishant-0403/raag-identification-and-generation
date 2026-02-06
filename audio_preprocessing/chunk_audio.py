import os
from pydub import AudioSegment

INPUT_DIR = "mp3_files"
OUTPUT_DIR = "chunks_45s"
CHUNK_LENGTH_MS = 45 * 1000  # 45 seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".mp3"):
        continue

    filepath = os.path.join(INPUT_DIR, filename)
    audio = AudioSegment.from_mp3(filepath)

    # Standardize audio
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(22050)

    duration_ms = len(audio)
    num_chunks = duration_ms // CHUNK_LENGTH_MS

    base = os.path.splitext(filename)[0]

    for i in range(num_chunks):
        start = i * CHUNK_LENGTH_MS
        end = start + CHUNK_LENGTH_MS

        chunk = audio[start:end]

        out_name = f"{base}_chunk{i:03d}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        chunk.export(out_path, format="wav")
