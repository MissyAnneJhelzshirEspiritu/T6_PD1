pip install numpy scipy scikit-learn librosa pyroomacoustics soundfile matplotlib
"""
Design of a Machine Learning-Based Reverberation Classification System
for Spatial Analysis Using Digital Twin

Run as a notebook or script. If running as script, make sure to adjust paths.
"""

import os
import numpy as np
import scipy.signal as sps
import soundfile as sf
import librosa
import pyroomacoustics as pra
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# -------------------------
# 1) Parameters & helpers
# -------------------------
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds of audio per example
N_CLASSES = 3   # Low, Medium, High reverberation
np.random.seed(42)

def save_wav(path, x, sr=SAMPLE_RATE):
    sf.write(path, x, sr)

# Schroeder T60 estimation (basic)
def estimate_t60_from_rir(rir, fs=SAMPLE_RATE):
    """
    Estimate T60 using Schroeder integration and linear fit on the decay curve.
    Returns T60 in seconds. If estimation fails returns np.nan.
    """
    rir = np.array(rir)
    energy = np.cumsum(rir[::-1]**2)[::-1]
    energy_db = 10 * np.log10(energy / np.max(energy) + 1e-10)
    # find region between -5 dB to -35 dB
    try:
        idx_start = np.where(energy_db <= -5)[0][0]
        idx_end   = np.where(energy_db <= -35)[0][0]
    except:
        return np.nan
    x = np.arange(idx_start, idx_end) / fs
    y = energy_db[idx_start:idx_end]
    # linear fit
    a, b = np.polyfit(x, y, 1)
    # slope a is dB/s. T60 is -60 / slope
    if a >= 0:
        return np.nan
    t60 = -60.0 / a
    return t60

# Feature extraction per reverberant audio sample
def extract_features(y, sr=SAMPLE_RATE, rir=None):
    feats = []
    # MFCC (mean + std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))
    # spectral centroid mean/std
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats.append(np.mean(sc))
    feats.append(np.std(sc))
    # spectral rolloff
    sr_off = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats.append(np.mean(sr_off))
    feats.append(np.std(sr_off))
    # zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    feats.append(np.mean(zcr))
    feats.append(np.std(zcr))
    # RMS energy
    rm = librosa.feature.rms(y=y)
    feats.append(np.mean(rm))
    feats.append(np.std(rm))
    # T60 estimate if RIR provided
    if rir is not None:
        t60 = estimate_t60_from_rir(rir, fs=sr)
        feats.append(0.0 if np.isnan(t60) else t60)
    else:
        feats.append(0.0)
    return np.array(feats, dtype=np.float32)


# -------------------------
# 2) Create digital twin rooms & RIRs (simulate)
# -------------------------
def simulate_room_rir(room_dim, source_loc, mic_loc, absorption=0.2, max_order=6, fs=SAMPLE_RATE):
    """
    Use pyroomacoustics to simulate an RIR for given room dimensions and positions.
    Returns RIR array (mono).
    absorption: wall absorption coefficient (0..1). Lower -> more reverberant.
    """
    # Convert absorption to reflection coefficient/material approximation using pyroomacoustics convenience
    e_absorption = absorption
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(source_loc)
    room.add_microphone_array(pra.MicrophoneArray(np.array(mic_loc).reshape(3,1), room.fs))
    room.compute_rir()
    rir = room.rir[0][0]  # single mic, single source
    return np.array(rir)

# -------------------------
# 3) Generate dataset
# -------------------------
def generate_dataset(num_examples_per_class=100, out_dir='data', clean_audio_files=None):
    """
    Generates synthetic dataset. For each class (Low/Med/High), creates examples by:
     - sampling room parameters
     - simulating RIR
     - convolving a clean sound with the RIR
    Returns X_feats, y_labels, and optionally saves audio to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    classes = ['low', 'medium', 'high']
    X = []
    y = []
    # If no clean_audio_files, generate white noise or a built-in librosa snippet
    if clean_audio_files is None:
        # use librosa example speech snippet repeated/padded
        snippet, _ = librosa.load(librosa.ex('trumpet'), sr=SAMPLE_RATE, mono=True)
        # cut or pad to DURATION
        target_len = int(DURATION * SAMPLE_RATE)
        if len(snippet) < target_len:
            snippet = np.tile(snippet, int(np.ceil(target_len / len(snippet))))[:target_len]
        else:
            snippet = snippet[:target_len]
        clean_list = [snippet]
    else:
        clean_list = []
        for f in clean_audio_files:
            y_c, _ = librosa.load(f, sr=SAMPLE_RATE, mono=True)
            # center crop/pad
            tlen = int(DURATION * SAMPLE_RATE)
            if len(y_c) < tlen:
                y_c = np.pad(y_c, (0, max(0, tlen - len(y_c))))
            y_c = y_c[:tlen]
            clean_list.append(y_c)

    for cls_idx, cls in enumerate(classes):
        print(f'Generating class {cls}...')
        for i in range(num_examples_per_class):
            # sample room size (meters)
            room_x = np.random.uniform(3.0, 10.0)
            room_y = np.random.uniform(3.0, 10.0)
            room_z = np.random.uniform(2.5, 4.0)
            room_dim = [room_x, room_y, room_z]
            # choose absorption based on class (low absorption -> high reverberation)
            if cls == 'low':
                absorption = np.random.uniform(0.6, 0.95)  # highly absorptive -> low reverb
            elif cls == 'medium':
                absorption = np.random.uniform(0.3, 0.6)
            else:
                absorption = np.random.uniform(0.01, 0.3)  # very reflective -> high reverb
            # random positions
            src = [np.random.uniform(0.5, room_x-0.5),
                   np.random.uniform(0.5, room_y-0.5),
                   np.random.uniform(1.0, 1.8)]
            mic = [np.random.uniform(0.5, room_x-0.5),
                   np.random.uniform(0.5, room_y-0.5),
                   np.random.uniform(1.0, 1.8)]
            rir = simulate_room_rir(room_dim, src, mic, absorption=absorption, max_order=8)
            # pick a clean audio
            clean = clean_list[np.random.randint(len(clean_list))]
            reverberant = sps.fftconvolve(clean, rir)[:len(clean)]
            # normalize
            reverberant = reverberant / (np.max(np.abs(reverberant)) + 1e-9) * 0.9
            # extract features (include the rir to get T60 estimate)
            feats = extract_features(reverberant, sr=SAMPLE_RATE, rir=rir)
            X.append(feats)
            y.append(cls_idx)
            # optionally save small subset of audio files
            if i < 3 and cls_idx == 0:
                save_wav(os.path.join(out_dir, f'{cls}_{i}.wav'), reverberant, SAMPLE_RATE)
    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)
    return X, y

# -------------------------
# 4) Train & evaluate
# -------------------------
def train_and_evaluate(X, y, save_model_path='reverb_rf.joblib'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['low','medium','high']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    joblib.dump(clf, save_model_path)
    print(f"Saved model to {save_model_path}")
    return clf

# -------------------------
# 5) Example usage (main)
# -------------------------
if __name__ == '__main__':
    # 1. Create dataset (this will take some time depending on numbers)
    X, y = generate_dataset(num_examples_per_class=150, out_dir='data_example')

    # 2. Train
    clf = train_and_evaluate(X, y, save_model_path='reverb_rf.joblib')

    # 3. Quick demonstration: load an audio file, estimate class
    # (Use first saved example)
    demo_file = os.path.join('data_example', 'low_0.wav')
    if os.path.exists(demo_file):
        y_demo, _ = librosa.load(demo_file, sr=SAMPLE_RATE, mono=True)
        # Extract features without providing rir (we might not have it) - T60 slot will be 0
        feats_demo = extract_features(y_demo, sr=SAMPLE_RATE, rir=None)
        pred = clf.predict(feats_demo.reshape(1,-1))[0]
        print("Predicted class for demo audio:", ['low','medium','high'][pred])

    # 4. Plot feature importance
    try:
        importances = clf.feature_importances_
        plt.figure(figsize=(10,4))
        plt.bar(range(len(importances)), importances)
        plt.title('Feature importances (Random Forest)')
        plt.xlabel('Feature index')
        plt.ylabel('Importance')
        plt.show()
    except Exception as e:
        print("Could not plot importances:", e)
