import librosa
import numpy as np
import soundfile as sf

# =========================
# LOAD AUDIO
# =========================
y, sr = librosa.load("audio.wav", sr=22050)
duration = len(y) / sr

# =========================
# TEMPO ESTIMATION
# =========================
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f"Estimated Tempo: {tempo:.2f} BPM")

# =========================
# PARAMETER ADAPTATION
# =========================
if tempo < 60:
    threshold_factor = 0.4
    min_gap = 1.0
elif tempo < 120:
    threshold_factor = 0.5
    min_gap = 0.7
else:
    threshold_factor = 0.6
    min_gap = 0.5

T_min = 3.0
T_max = 10.0

# =========================
# ENERGY
# =========================
frame_length = int(0.025 * sr)
hop_length = int(0.010 * sr)

energy = librosa.feature.rms(
    y=y,
    frame_length=frame_length,
    hop_length=hop_length
)[0]

energy_smooth = np.convolve(energy, np.ones(5)/5, mode='same')
threshold = threshold_factor * np.mean(energy_smooth)

# =========================
# LOW ENERGY BOUNDARIES
# =========================
low_idx = np.where(energy_smooth < threshold)[0]
times = librosa.frames_to_time(low_idx, sr=sr, hop_length=hop_length)

filtered = []
last = -np.inf

for t in times:
    if t - last > min_gap:
        filtered.append(t)
        last = t

filtered = [0.0] + filtered + [duration]

# =========================
# INITIAL CHUNKS (NO OVERLAP)
# =========================
chunks = []
for i in range(len(filtered)-1):
    s, e = filtered[i], filtered[i+1]
    if e > s:
        chunks.append((s, e))

# =========================
# PITCH EXTRACTION
# =========================
f0, _, _ = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

f0_clean = np.nan_to_num(f0)
f0_smooth = np.convolve(f0_clean, np.ones(5)/5, mode='same')

# =========================
# FIND SPLIT AFTER NYAS
# =========================
def find_split_after_stable(start, end):
    s_frame = int(start * sr / hop_length)
    e_frame = int(end * sr / hop_length)

    segment = f0_smooth[s_frame:e_frame]

    if len(segment) < 5:
        return None

    delta_pitch = np.abs(np.diff(segment))
    stable = delta_pitch < 5

    regions = []
    cur = None

    for i, val in enumerate(stable):
        if val and cur is None:
            cur = i
        elif not val and cur is not None:
            regions.append((cur, i))
            cur = None

    if cur is not None:
        regions.append((cur, len(stable)))

    min_len = int(0.3 / (hop_length / sr))
    long_regions = [(s, e) for s, e in regions if (e - s) >= min_len]

    if not long_regions:
        return None

    best = max(long_regions, key=lambda r: r[1])

    s_idx, e_idx = best

    split_frame = e_idx
    split_time = start + (split_frame * hop_length / sr)

    if split_time >= end:
        return None

    return split_time

# =========================
# SPLITTING WITH PITCH
# =========================
split_chunks = []

for s, e in chunks:
    current = s

    while current < e:
        max_end = min(current + T_max, e)

        split_point = find_split_after_stable(current, max_end)

        if split_point is not None and split_point > current:
            split_chunks.append((current, split_point))
            current = split_point
        else:
            split_chunks.append((current, max_end))
            current = max_end

# =========================
# ENFORCE MIN LENGTH
# =========================
def enforce_min_length(chunks, T_min):
    merged = chunks.copy()

    while True:
        new_chunks = []
        changed = False
        i = 0

        while i < len(merged):
            s, e = merged[i]
            if e - s < T_min:
                changed = True
                if i + 1 < len(merged):
                    ns, ne = merged[i+1]
                    new_chunks.append((s, ne))
                    i += 2
                else:
                    if new_chunks:
                        ps, pe = new_chunks[-1]
                        new_chunks[-1] = (ps, e)
                    else:
                        new_chunks.append((s, e))
                    i += 1
            else:
                new_chunks.append((s, e))
                i += 1

        if not changed:
            break

        merged = new_chunks

    return merged

final_chunks = enforce_min_length(split_chunks, T_min)

# =========================
# EXTRACT AUDIO
# =========================
audio_chunks = []

for s, e in final_chunks:
    start_sample = int(s * sr)
    end_sample = int(e * sr)

    if end_sample > start_sample:
        audio_chunks.append(y[start_sample:end_sample])

# =========================
# SAVE
# =========================
for i, chunk in enumerate(audio_chunks):
    sf.write(f"chunk_{i}.wav", chunk, sr)

# =========================
# SUMMARY
# =========================
print(f"\nTotal chunks: {len(audio_chunks)}\n")

for i, (s, e) in enumerate(final_chunks):
    print(f"Chunk {i}: {s:.2f}s → {e:.2f}s | Length: {e-s:.2f}s")
