#!/usr/bin/env python3
"""Widget interativo para gravar áudio do ESP32-S3 (`grab`) e testar parâmetros de VAD, NS e k-NN."""
import base64
import glob
import sys
import time
import wave
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import numpy as np
import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recvoz.features import (  # noqa: E402
    N_FFT,
    N_MFCC,
    TARGET_SR,
    VAD_CHUNK,
    WIN_SAMPLES,
    bandpass,
    load_audio,
    load_dataset,
    noise_profile,
    power,
)

MEL_FB = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=40, htk=True)


def find_ports():
    return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def grab_from_esp(port, seconds=3, baud=115200):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.5
    ser.dtr = False
    ser.rts = False
    ser.open()
    try:
        # No Linux/CP210x abrir a porta pode reiniciar o ESP32 (~2 s de boot).
        boot_deadline = time.monotonic() + 3.0
        while time.monotonic() < boot_deadline:
            raw = ser.readline().decode("ascii", errors="ignore").strip()
            if "recvoz pronto" in raw or "recvoz>" in raw:
                break
        time.sleep(0.15)
        ser.reset_input_buffer()
        cmd = f"grab {seconds}\n".encode()
        ser.write(cmd)
        ser.flush()

        deadline = time.monotonic() + 30.0
        retry_at = time.monotonic() + 4.0
        started = False
        b64_lines = []
        seen = []
        while time.monotonic() < deadline:
            raw = ser.readline().decode("ascii", errors="ignore").strip()
            if not started and time.monotonic() > retry_at:
                ser.write(cmd)
                ser.flush()
                retry_at = time.monotonic() + 4.0
            if not raw:
                continue
            seen.append(raw)
            if len(seen) > 12:
                seen.pop(0)
            if "Unrecognized command" in raw:
                raise RuntimeError("O firmware na placa ainda não tem o comando `grab`. Rode `idf.py flash` primeiro.")
            if raw.startswith("RECVOZ_B64_START"):
                started = True
                continue
            if raw == "RECVOZ_B64_END":
                break
            if started:
                b64_lines.append(raw)
        if not started or not b64_lines:
            tail = "\n".join(seen[-6:]) if seen else "(nenhum dado recebido nessa porta)"
            raise RuntimeError(
                f"ESP32 não respondeu ao `grab` em {port}.\n"
                f"Últimas linhas recebidas:\n{tail}"
            )
        pcm_bytes = base64.b64decode("".join(b64_lines))
        return np.frombuffer(pcm_bytes, dtype="<i2").copy()
    finally:
        ser.close()


def save_wav(path, pcm):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm.astype("<i2").tobytes())


def extract_mfcc(pcm, noise=None, mode="pre_mel", alpha=1.2, floor=0.05):
    y = np.asarray(pcm, dtype=np.float64) / 32768.0
    p = power(bandpass(y))
    if noise is not None and mode == "pre_mel":
        p = np.maximum(p - alpha * noise[:, None], floor * p)
    mel = MEL_FB @ p
    if noise is not None and mode == "mel_wiener":
        mel_n = (MEL_FB @ noise)[:, None]
        pos = mel_n > 0.0
        if pos.any():
            snr = np.where(pos, np.maximum(mel / np.where(pos, mel_n, 1.0) - 1.0, 0.0), 0.0)
            mel = np.where(pos, np.maximum(snr / (snr + alpha), floor), 1.0) * mel
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max(axis=0, keepdims=True) - 80.0)
    m = librosa.feature.mfcc(S=db, sr=TARGET_SR, n_mfcc=N_MFCC)
    return np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)])


def frame_mod(m39):
    return float(np.linalg.norm(5.0 * np.sqrt(np.mean(m39[14:26] ** 2, axis=1))))


def cos_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64)[1:], np.asarray(b, dtype=np.float64)[1:]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 1.0
    return 1.0 - float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def classify_vec(query, vecs, labels, k=3, knn_mode="lmknn"):
    scores = {}
    for spk in sorted(set(labels)):
        dists = sorted(cos_dist(query, v) for v, l in zip(vecs, labels) if l == spk)
        if knn_mode == "lmknn":
            scores[spk] = float(np.mean(dists[:min(k, len(dists))]))
        else:
            scores[spk] = dists[0]
    if knn_mode == "global_1d":
        all_d = sorted((cos_dist(query, v), l) for v, l in zip(vecs, labels))[:k]
        w_map = {}
        for d, l in all_d:
            w_map[l] = w_map.get(l, 0.0) + 1.0 / (d + 1e-9)
        ranked = sorted(scores.items(), key=lambda x: x[1])
        win_lbl = max(w_map, key=w_map.get)
        return win_lbl, ranked
    ranked = sorted(scores.items(), key=lambda x: x[1])
    return ranked[0][0], ranked


class TunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RecVoz — Laboratório Interativo (ESP32-S3)")
        self.root.geometry("1080x720")

        self.ds = load_dataset(ROOT / "audio_exemplo", threshold=300)
        self.extra_vecs = []
        self.extra_labels = []
        self.silence_pcm = None
        self.test_pcm = None
        self.silence_rms = 0

        self._build_ui()
        self._load_default_sample()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Porta Serial:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value=(find_ports() or ["/dev/ttyUSB0"])[0])
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var, values=find_ports(), width=16)
        self.port_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Atualizar Portas", command=self._refresh_ports).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Gravar Silêncio (2s)", command=lambda: self._grab("silence", 2)).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Gravar Fala Teste (3s)", command=lambda: self._grab("test", 3)).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Abrir WAV Teste...", command=self._open_wav).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Cadastrar Teste como VISITANTE", command=self._enroll_visitor).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Limpar VISITANTE", command=self._clear_visitor).pack(side=tk.LEFT, padx=2)

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.LabelFrame(main, text="Parâmetros em Tempo Real", padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        self.vad_var = tk.IntVar(value=300)
        self.tau_var = tk.DoubleVar(value=0.042)
        self.ns_on_var = tk.BooleanVar(value=True)
        self.ns_mode_var = tk.StringVar(value="pre_mel")
        self.alpha_var = tk.DoubleVar(value=1.2)
        self.floor_var = tk.DoubleVar(value=0.05)
        self.k_var = tk.IntVar(value=3)
        self.knn_mode_var = tk.StringVar(value="lmknn")
        self.mod_var = tk.DoubleVar(value=32.0)

        self._add_slider(ctrl, "VAD (RMS 80-7920Hz)", self.vad_var, 100, 1200, 10)
        self._add_slider(ctrl, "Tau (limiar)", self.tau_var, 0.010, 0.120, 0.002)
        ttk.Checkbutton(ctrl, text="NS (Supressão de Ruído)", variable=self.ns_on_var, command=self.recompute).pack(anchor=tk.W, pady=4)
        ttk.Label(ctrl, text="Modo NS:").pack(anchor=tk.W)
        ttk.Combobox(ctrl, textvariable=self.ns_mode_var, values=["pre_mel", "mel_wiener"], state="readonly").pack(fill=tk.X, pady=(0, 6))
        self.ns_mode_var.trace_add("write", lambda *_: self.recompute())
        self._add_slider(ctrl, "NS Alpha / Over", self.alpha_var, 0.2, 3.0, 0.1)
        self._add_slider(ctrl, "NS Floor", self.floor_var, 0.01, 0.30, 0.01)
        self._add_slider(ctrl, "k (vizinhos)", self.k_var, 1, 7, 1)
        ttk.Label(ctrl, text="Modo k-NN:").pack(anchor=tk.W)
        ttk.Combobox(ctrl, textvariable=self.knn_mode_var, values=["lmknn", "global_1d"], state="readonly").pack(fill=tk.X, pady=(0, 6))
        self.knn_mode_var.trace_add("write", lambda *_: self.recompute())
        self._add_slider(ctrl, "Min Modulação (voz)", self.mod_var, 15.0, 50.0, 1.0)

        self.status_lbl = tk.Label(ctrl, text="", justify=tk.LEFT, font=("Monospace", 10), bg="#111", fg="#eee", padx=8, pady=8)
        self.status_lbl.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_rms = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_slider(self, parent, label, var, lo, hi, res):
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        s = tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL, length=240, command=lambda _: self.recompute())
        s.pack(fill=tk.X, pady=(0, 4))

    def _refresh_ports(self):
        ports = find_ports()
        self.port_combo["values"] = ports
        if ports:
            self.port_var.set(ports[0])

    def _grab(self, kind, sec):
        port = self.port_var.get().strip()
        try:
            self.status_lbl.config(text=f"Gravando {sec}s de {port}...\n(aguarde envio serial)")
            self.root.update_idletasks()
            pcm = grab_from_esp(port, seconds=sec)
            out = ROOT / f"scratch_esp_{kind}.wav"
            save_wav(out, pcm)
            if kind == "silence":
                self.silence_pcm = pcm[:8192]
                y = bandpass(self.silence_pcm.astype(np.float64) / 32768.0)
                self.silence_rms = int(np.sqrt(np.mean((y * 32768.0) ** 2)))
            else:
                self.test_pcm = pcm
            self.recompute()
        except Exception as exc:
            messagebox.showerror("Erro Serial", str(exc))

    def _open_wav(self):
        p = filedialog.askopenfilename(initialdir=ROOT, filetypes=[("WAV", "*.wav")])
        if p:
            self.test_pcm = load_audio(p)
            self.recompute()

    def _load_default_sample(self):
        self.test_pcm = load_audio(ROOT / "audio_exemplo/locutor_adrian_s1.wav")[:48000]
        self.recompute()

    def _enroll_visitor(self):
        if self.test_pcm is None:
            return
        nz = self._current_noise()
        m = extract_mfcc(self.test_pcm[:WIN_SAMPLES] if len(self.test_pcm) >= WIN_SAMPLES else np.pad(self.test_pcm, (0, WIN_SAMPLES - len(self.test_pcm))),
                         noise=nz, mode=self.ns_mode_var.get(), alpha=self.alpha_var.get(), floor=self.floor_var.get())
        self.extra_vecs.append(m.mean(axis=1))
        self.extra_labels.append("VISITANTE")
        self.recompute()

    def _clear_visitor(self):
        self.extra_vecs.clear()
        self.extra_labels.clear()
        self.recompute()

    def _current_noise(self):
        if not self.ns_on_var.get() or self.silence_pcm is None:
            return None
        return noise_profile(self.silence_pcm[:8192])

    def recompute(self):
        if self.test_pcm is None:
            return
        pcm = self.test_pcm
        chunks = [pcm[i:i + VAD_CHUNK] for i in range(0, len(pcm) - VAD_CHUNK + 1, VAD_CHUNK)]
        rms_list = [int(np.sqrt(np.mean((bandpass(c.astype(np.float64) / 32768.0) * 32768.0) ** 2))) for c in chunks]
        eff_vad = max(self.vad_var.get(), 2 * self.silence_rms) if (self.ns_on_var.get() and self.silence_pcm is not None) else self.vad_var.get()
        voiced_chunks = [c for c, r in zip(chunks, rms_list) if r > eff_vad]

        nz = self._current_noise()
        mode = self.ns_mode_var.get()
        alpha = self.alpha_var.get()
        floor = self.floor_var.get()

        if len(voiced_chunks) == 0:
            win = pcm[:WIN_SAMPLES] if len(pcm) >= WIN_SAMPLES else np.pad(pcm, (0, WIN_SAMPLES - len(pcm)))
            vad_info = f"0/{len(chunks)} blocos > {eff_vad} (fallback janela crua)"
        else:
            joined = np.concatenate(voiced_chunks)
            win = joined[:WIN_SAMPLES] if len(joined) >= WIN_SAMPLES else np.pad(joined, (0, WIN_SAMPLES - len(joined)))
            vad_info = f"{len(voiced_chunks)}/{len(chunks)} blocos > {eff_vad}"

        m39 = extract_mfcc(win, noise=nz, mode=mode, alpha=alpha, floor=floor)
        qvec = m39.mean(axis=1)
        vmod = frame_mod(m39)

        d_ns = 1.0
        if nz is not None:
            anc = extract_mfcc(self.silence_pcm[:8192], noise=nz, mode=mode, alpha=alpha, floor=floor).mean(axis=1)
            d_ns = cos_dist(qvec, anc)

        all_vecs = list(self.ds.vectors) + self.extra_vecs
        all_labels = list(self.ds.labels) + self.extra_labels
        win_lbl, ranked = classify_vec(qvec, all_vecs, all_labels, k=self.k_var.get(), knn_mode=self.knn_mode_var.get())

        d_min = ranked[0][1]
        is_noise = (vmod < self.mod_var.get()) or (d_ns <= d_min)
        tau = self.tau_var.get()
        if is_noise:
            decision = "RUIDO AMBIENTE"
        elif d_min > tau:
            decision = f"DESCONHECIDO (quase {win_lbl})"
        else:
            decision = f"CONHECIDO: {win_lbl}"

        lines = [
            f"Decisão : {decision}",
            f"VAD     : {vad_info} (sil_rms={self.silence_rms})",
            f"Mod/NS  : mod={vmod:.1f} (<{self.mod_var.get():.0f})  d_ns={d_ns:.4f}",
            f"Tau     : {tau:.3f}",
            "--- Ranking ---",
        ]
        for spk, dist in ranked:
            lines.append(f"  {spk:<10s}: {dist:.4f}")
        self.status_lbl.config(text="\n".join(lines))

        self.ax_wave.clear()
        self.ax_wave.plot(np.arange(len(pcm)) / TARGET_SR, pcm, lw=0.6)
        self.ax_wave.set_title("Áudio capturado (PCM int16)")
        self.ax_wave.set_ylabel("Amplitude")

        self.ax_rms.clear()
        t_rms = np.arange(len(rms_list)) * 0.1
        colors = ["#2ca02c" if r > eff_vad else "#888888" for r in rms_list]
        self.ax_rms.bar(t_rms, rms_list, width=0.08, color=colors)
        self.ax_rms.axhline(eff_vad, color="r", ls="--", label=f"VAD efetivo ({eff_vad})")
        self.ax_rms.legend(loc="upper right")
        self.ax_rms.set_title("RMS 80-7920 Hz por bloco (100 ms)")
        self.ax_rms.set_xlabel("Tempo (s)")
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    TunerApp(root)
    root.mainloop()
