#!/usr/bin/env python3
"""Interface gráfica (Tkinter) para coletar o dataset RecVoz direto do ESP32-S3."""
import glob
import re
import shutil
import subprocess
import sys
import threading
import unicodedata
from pathlib import Path
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # ambiente headless/CI sem python3-tk
    tk = None
    messagebox = None
    ttk = None

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from grab_esp import open_serial, plan_chunks, record_chunks  # noqa: E402
from recvoz.features import auto_noise_profile, load_audio, voiced_windows  # noqa: E402

GROUPS = [
    ("known", "Conhecido", ROOT / "audio_exemplo"),
    ("cal", "Impostor (Calibração)", ROOT / "dados" / "desconhecidos_cal"),
    ("test", "Impostor (Teste)", ROOT / "dados" / "desconhecidos_teste"),
    ("noise", "Ruído / Silêncio", ROOT / "dados" / "ruido"),
]

PROMPTS = [
    (
        "O sol nasceu claro atrás das montanhas enquanto os pássaros começavam a cantar "
        "no jardim. A brisa fresca atravessava as janelas abertas, trazendo o cheiro "
        "de café recém-passado pela cozinha inteira."
    ),
    (
        "Durante a viagem de trem pelo interior, observamos pequenas estações antigas, "
        "pontes de ferro sobre rios tranquilos e campos verdes que mudavam de cor "
        "conforme as nuvens passavam lentamente pelo céu."
    ),
    (
        "A tecnologia de áudio embarcado exige equilíbrio entre precisão matemática e "
        "baixo consumo de energia. Pequenos microcontroladores conseguem processar "
        "sinais de voz em tempo real sem depender de servidores externos."
    ),
    (
        "Na feira de domingo, as barracas exibiam frutas tropicais, queijos artesanais "
        "e flores coloridas. Os vendedores anunciavam os preços com entusiasmo enquanto "
        "os moradores conversavam calmamente pelas calçadas."
    ),
    (
        "Quando a chuva forte chegou no fim da tarde, as luzes da cidade refletiam nas "
        "ruas molhadas. Dentro da biblioteca silenciosa, estudantes revisavam anotações "
        "e organizavam livros sobre a mesa de madeira."
    ),
    (
        "Antes de começar qualquer experimento em laboratório, é essencial conferir as "
        "conexões elétricas, anotar as condições do ambiente e verificar se os sensores "
        "estão calibrados corretamente para garantir resultados confiáveis."
    ),
]


def find_ports():
    return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def sanitize_name(raw):
    norm = unicodedata.normalize("NFKD", raw.strip()).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", norm).lower()
    return cleaned[:15]


def next_session_number(folder, speaker_name):
    if not speaker_name or not folder.exists():
        return 1
    pattern = re.compile(rf"^locutor_{re.escape(speaker_name)}_s(\d+)\.wav$", re.IGNORECASE)
    found = []
    for p in folder.glob("locutor_*.wav"):
        m = pattern.match(p.name)
        if m:
            found.append(int(m.group(1)))
    return max(found, default=0) + 1


def build_output_path(group_key, raw_name, session_num):
    folder = dict((k, d) for k, _, d in GROUPS)[group_key]
    name = sanitize_name(raw_name)
    if not name:
        return None
    if group_key == "noise":
        return folder / f"{name}_s{session_num}.wav"
    return folder / f"locutor_{name}_s{session_num}.wav"


def analyze_wav(path, vad=300):
    pcm = load_audio(path)
    duration = len(pcm) / 16000.0
    peak = float(np.max(np.abs(pcm))) / 32768.0 if len(pcm) else 0.0
    windows = list(voiced_windows(pcm, threshold=vad))
    noise = auto_noise_profile(pcm, threshold=vad)
    return {
        "duration": duration,
        "peak": peak,
        "windows": len(windows),
        "has_auto_noise": noise is not None,
    }


def play_wav(path):
    for cmd in (["paplay", str(path)], ["aplay", "-q", str(path)], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]):
        if shutil.which(cmd[0]):
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    return False


class ColetorApp(tk.Tk if tk is not None else object):
    def __init__(self):
        super().__init__()
        self.title("RecVoz — Coletor de Dataset (ESP32-S3)")
        self.geometry("860x720")
        self.minsize(780, 640)

        self.recording = False
        self.last_saved = None
        self.prompt_idx = 0

        self.port_var = tk.StringVar()
        self.baud_var = tk.IntVar(value=115200)
        self.dur_var = tk.IntVar(value=18)
        self.group_var = tk.StringVar(value="known")
        self.name_var = tk.StringVar(value="adrian")
        self.session_var = tk.IntVar(value=1)
        self.preview_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Pronto para gravar.")
        self.quality_var = tk.StringVar(value="Nenhuma gravação realizada nesta sessão.")
        self.summary_var = tk.StringVar()

        self._build_ui()
        self._refresh_ports()
        self._on_identity_change()
        self._refresh_dataset_tree()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # 1. Conexão
        conn = ttk.LabelFrame(self, text="1. Conexão ESP32-S3 (firmware recvoz-esp)")
        conn.pack(fill=tk.X, **pad)

        ttk.Label(conn, text="Porta:").pack(side=tk.LEFT, padx=(8, 4), pady=6)
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var, width=22, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(conn, text="Atualizar", command=self._refresh_ports).pack(side=tk.LEFT, padx=4, pady=6)

        ttk.Label(conn, text="Duração (s):").pack(side=tk.LEFT, padx=(18, 4), pady=6)
        ttk.Spinbox(conn, from_=3, to=60, increment=3, textvariable=self.dur_var, width=5).pack(side=tk.LEFT, padx=4, pady=6)
        for preset in (10, 18, 24):
            ttk.Button(conn, text=f"{preset}s", width=4, command=lambda s=preset: self.dur_var.set(s)).pack(side=tk.LEFT, padx=2, pady=6)

        # 2. Grupo e Identificação
        meta = ttk.LabelFrame(self, text="2. Destino e Locutor")
        meta.pack(fill=tk.X, **pad)

        grp_row = ttk.Frame(meta)
        grp_row.pack(fill=tk.X, padx=8, pady=(6, 4))
        for key, label, _ in GROUPS:
            ttk.Radiobutton(
                grp_row, text=label, value=key, variable=self.group_var, command=self._on_identity_change
            ).pack(side=tk.LEFT, padx=(0, 14))

        id_row = ttk.Frame(meta)
        id_row.pack(fill=tk.X, padx=8, pady=(4, 6))
        ttk.Label(id_row, text="Nome (sem acento):").pack(side=tk.LEFT)
        name_entry = ttk.Entry(id_row, textvariable=self.name_var, width=20)
        name_entry.pack(side=tk.LEFT, padx=(6, 14))
        name_entry.bind("<KeyRelease>", lambda _e: self._on_identity_change())

        ttk.Label(id_row, text="Sessão s:").pack(side=tk.LEFT)
        sess_spin = ttk.Spinbox(id_row, from_=1, to=99, textvariable=self.session_var, width=4, command=self._update_preview)
        sess_spin.pack(side=tk.LEFT, padx=(4, 10))
        sess_spin.bind("<KeyRelease>", lambda _e: self._update_preview())
        ttk.Button(id_row, text="Auto sN", command=self._auto_session).pack(side=tk.LEFT, padx=4)

        ttk.Label(meta, textvariable=self.preview_var, foreground="#1a4d8f").pack(anchor=tk.W, padx=8, pady=(0, 6))

        # 3. Teleprompter
        prompt_box = ttk.LabelFrame(self, text="3. Texto para leitura (espere ~1 s em silêncio no início e leia normalmente)")
        prompt_box.pack(fill=tk.X, **pad)

        self.prompt_text = tk.Text(prompt_box, height=4, wrap=tk.WORD, font=("TkDefaultFont", 12), bg="#fffdf5", relief=tk.FLAT, padx=10, pady=8)
        self.prompt_text.pack(fill=tk.X, padx=8, pady=(6, 4))
        self._set_prompt(0)

        p_btns = ttk.Frame(prompt_box)
        p_btns.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Button(p_btns, text="Trocar texto", command=self._next_prompt).pack(side=tk.LEFT)
        ttk.Label(
            p_btns,
            text="Dica: varie o texto e um pouco a posição entre s1, s2 e s3 do mesmo locutor.",
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=12)

        # 4. Controles de Gravação
        rec_box = ttk.LabelFrame(self, text="4. Captura e Checagem Automática")
        rec_box.pack(fill=tk.X, **pad)

        btn_row = ttk.Frame(rec_box)
        btn_row.pack(fill=tk.X, padx=8, pady=8)

        self.rec_btn = tk.Button(
            btn_row,
            text="●  GRAVAR",
            command=lambda: self._start_record(overwrite=False),
            bg="#c62828",
            fg="white",
            activebackground="#a31515",
            activeforeground="white",
            font=("TkDefaultFont", 12, "bold"),
            padx=20,
            pady=6,
            cursor="hand2",
        )
        self.rec_btn.pack(side=tk.LEFT)

        self.rerec_btn = ttk.Button(btn_row, text="Regravar último", command=lambda: self._start_record(overwrite=True), state=tk.DISABLED)
        self.rerec_btn.pack(side=tk.LEFT, padx=8)

        self.play_btn = ttk.Button(btn_row, text="Ouvir último", command=self._play_last, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=4)

        self.prog = ttk.Progressbar(btn_row, mode="determinate", length=260)
        self.prog.pack(side=tk.RIGHT, padx=4)

        ttk.Label(rec_box, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=8, pady=(0, 2))
        self.quality_lbl = ttk.Label(rec_box, textvariable=self.quality_var, foreground="#333333")
        self.quality_lbl.pack(anchor=tk.W, padx=8, pady=(0, 8))

        # 5. Progresso do Dataset
        ds_box = ttk.LabelFrame(self, text="5. Dataset Atual no Disco")
        ds_box.pack(fill=tk.BOTH, expand=True, **pad)

        top_ds = ttk.Frame(ds_box)
        top_ds.pack(fill=tk.X, padx=8, pady=(6, 4))
        ttk.Label(top_ds, textvariable=self.summary_var, font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        ttk.Button(top_ds, text="Recarregar lista", command=self._refresh_dataset_tree).pack(side=tk.RIGHT)

        cols = ("grupo", "arquivo", "dur", "janelas", "ruido")
        self.tree = ttk.Treeview(ds_box, columns=cols, show="headings", height=8)
        self.tree.heading("grupo", text="Grupo")
        self.tree.heading("arquivo", text="Arquivo")
        self.tree.heading("dur", text="Duração")
        self.tree.heading("janelas", text="Janelas VAD")
        self.tree.heading("ruido", text="Auto-Ruído")
        self.tree.column("grupo", width=140, anchor=tk.W)
        self.tree.column("arquivo", width=330, anchor=tk.W)
        self.tree.column("dur", width=80, anchor=tk.CENTER)
        self.tree.column("janelas", width=95, anchor=tk.CENTER)
        self.tree.column("ruido", width=95, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _refresh_ports(self):
        ports = find_ports()
        self.port_combo["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        elif not ports:
            self.port_var.set("")

    def _set_prompt(self, idx):
        self.prompt_idx = idx % len(PROMPTS)
        self.prompt_text.config(state=tk.NORMAL)
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", PROMPTS[self.prompt_idx])
        self.prompt_text.config(state=tk.DISABLED)

    def _next_prompt(self):
        self._set_prompt(self.prompt_idx + 1)

    def _auto_session(self):
        group_key = self.group_var.get()
        folder = dict((k, d) for k, _, d in GROUPS)[group_key]
        name = sanitize_name(self.name_var.get())
        if group_key == "noise":
            if folder.exists():
                existing = sorted(folder.glob(f"{name}_s*.wav")) if name else []
                self.session_var.set(len(existing) + 1)
            else:
                self.session_var.set(1)
        else:
            self.session_var.set(next_session_number(folder, name))
        self._update_preview()

    def _on_identity_change(self):
        self._auto_session()

    def _update_preview(self):
        try:
            sess = int(self.session_var.get())
        except (ValueError, tk.TclError):
            sess = 1
        out = build_output_path(self.group_var.get(), self.name_var.get(), sess)
        if out is None:
            self.preview_var.set("Arquivo de saída: (informe um nome válido)")
        else:
            rel = out.relative_to(ROOT)
            exists_tag = "  [JÁ EXISTE]" if out.exists() else ""
            self.preview_var.set(f"Arquivo de saída: {rel}{exists_tag}")

    def _set_busy(self, busy):
        self.recording = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.rec_btn.config(state=state, text="GRAVANDO..." if busy else "●  GRAVAR")
        self.rerec_btn.config(state=tk.DISABLED if (busy or not self.last_saved) else tk.NORMAL)
        self.play_btn.config(state=tk.DISABLED if (busy or not self.last_saved) else tk.NORMAL)

    def _start_record(self, overwrite=False):
        if self.recording:
            return
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Porta não encontrada", "Conecte o ESP32-S3 e clique em 'Atualizar'.")
            return

        if overwrite and self.last_saved is not None:
            out_path = self.last_saved
        else:
            try:
                sess = int(self.session_var.get())
            except (ValueError, tk.TclError):
                messagebox.showerror("Sessão inválida", "Informe um número de sessão inteiro (>= 1).")
                return
            out_path = build_output_path(self.group_var.get(), self.name_var.get(), sess)
            if out_path is None:
                messagebox.showerror("Nome inválido", "Informe um nome de locutor válido (letras/números).")
                return
            if out_path.exists():
                if not messagebox.askyesno("Sobrescrever?", f"O arquivo {out_path.name} já existe. Deseja sobrescrever?"):
                    return

        try:
            dur = int(self.dur_var.get())
            chunks = plan_chunks(dur)
        except Exception as exc:
            messagebox.showerror("Duração inválida", str(exc))
            return

        self.prog["maximum"] = len(chunks)
        self.prog["value"] = 0
        self._set_busy(True)
        self.status_var.set(f"Conectando em {port}...")
        self.quality_var.set("Aguardando captura...")

        thread = threading.Thread(
            target=self._record_worker,
            args=(port, self.baud_var.get(), dur, out_path, self.group_var.get(), overwrite),
            daemon=True,
        )
        thread.start()

    def _record_worker(self, port, baud, dur, out_path, group_key, overwrite):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(".partial.wav")

        def on_chunk(idx, total, sec):
            self.after(0, lambda i=idx, t=total, s=sec: self._on_progress(i, t, s))

        try:
            ser = open_serial(port, baud)
            try:
                record_chunks(ser, partial, dur, on_chunk=on_chunk)
            finally:
                ser.close()
            partial.replace(out_path)
            stats = analyze_wav(out_path)
            self.after(0, lambda: self._on_record_done(out_path, stats, group_key, overwrite))
        except Exception as exc:
            self.after(0, lambda e=exc: self._on_record_error(e))

    def _on_progress(self, idx, total, sec):
        self.prog["value"] = idx - 1
        self.status_var.set(f"[{idx}/{total}] Gravando {sec}s... FALE NORMALMENTE (olhe 'GRAVANDO' no OLED)")

    def _on_record_done(self, out_path, stats, group_key, overwrite):
        self.prog["value"] = self.prog["maximum"]
        self.last_saved = out_path
        self._set_busy(False)
        self.status_var.set(f"Salvo: {out_path.relative_to(ROOT)}")

        warns = []
        if group_key != "noise":
            if stats["windows"] < 5:
                warns.append(f"poucas janelas ({stats['windows']}; ideal >= 5)")
            if not stats["has_auto_noise"]:
                warns.append("sem pausa silenciosa p/ auto-ruído")
        if stats["peak"] >= 0.98:
            warns.append("possível saturação/clipping")

        noise_txt = "OK" if stats["has_auto_noise"] else "sem pausa"
        base_msg = (
            f"{stats['duration']:.1f}s | {stats['windows']} janelas válidas | "
            f"pico={stats['peak']:.2f} | auto-ruído: {noise_txt}"
        )
        if warns:
            self.quality_var.set(f"⚠ ATENÇÃO: {base_msg} — {', '.join(warns)}")
            self.quality_lbl.config(foreground="#b35c00")
        else:
            self.quality_var.set(f"✓ OK: {base_msg}")
            self.quality_lbl.config(foreground="#1b6e2d")

        if not overwrite:
            self._auto_session()
            self._next_prompt()
        self._update_preview()
        self._refresh_dataset_tree()

    def _on_record_error(self, exc):
        self._set_busy(False)
        self.status_var.set("Erro na gravação.")
        self.quality_var.set(str(exc))
        self.quality_lbl.config(foreground="#c62828")
        messagebox.showerror("Erro ao gravar", str(exc))

    def _play_last(self):
        if self.last_saved and self.last_saved.exists():
            if not play_wav(self.last_saved):
                messagebox.showwarning("Player indisponível", "Instale aplay, paplay ou ffplay para reproduzir.")

    def _refresh_dataset_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        counts = {"known": set(), "cal": set(), "test": set(), "noise": 0}
        known_sessions = {}

        for key, label, folder in GROUPS:
            if not folder.exists():
                continue
            for p in sorted(folder.glob("*.wav")):
                if p.name.endswith(".partial.wav"):
                    continue
                try:
                    st = analyze_wav(p)
                    dur_s = f"{st['duration']:.1f}s"
                    win_s = str(st["windows"])
                    ns_s = "OK" if st["has_auto_noise"] else "—"
                except Exception:
                    dur_s, win_s, ns_s = "?", "?", "?"
                self.tree.insert("", tk.END, values=(label, p.name, dur_s, win_s, ns_s))
                m = re.match(r"^locutor_([a-zA-Z0-9]+)_s\d+\.wav$", p.name)
                if key == "noise":
                    counts["noise"] += 1
                elif m:
                    spk = m.group(1).lower()
                    counts[key].add(spk)
                    if key == "known":
                        known_sessions[spk] = known_sessions.get(spk, 0) + 1

        ready_known = sum(1 for n in known_sessions.values() if n >= 3)
        self.summary_var.set(
            f"Conhecidos (>=3 sessões): {ready_known}/{len(known_sessions)}  |  "
            f"Impostores Cal: {len(counts['cal'])} (meta 2+)  |  "
            f"Impostores Teste: {len(counts['test'])} (meta 3+)  |  "
            f"Ruído: {counts['noise']}"
        )


def main():
    if tk is None:
        raise SystemExit("Tkinter não está instalado neste Python (instale python3-tk).")
    app = ColetorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
