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

# Textos com ~90-105 palavras (~35-45 s de leitura natural), suficientes para 18s-30s sem faltar texto.
PROMPTS = [
    (
        "O sol nasceu claro atrás das montanhas enquanto os pássaros começavam a cantar "
        "no jardim da casa antiga. A brisa fresca atravessava as janelas abertas, trazendo "
        "o cheiro de café recém-passado pela cozinha inteira. Na rua ainda tranquila, um "
        "ciclista subia devagar a ladeira de paralelepípedos, desviando das folhas secas "
        "acumuladas junto ao meio-fio. Dentro da padaria da esquina, o padeiro organizava "
        "as fornadas quentes de pão francês e broas de milho, cumprimentando os primeiros "
        "vizinhos que chegavam para começar a rotina de mais uma manhã de trabalho."
    ),
    (
        "Durante a longa viagem de trem pelo interior, observamos pequenas estações "
        "históricas, pontes de ferro sobre rios tranquilos e campos verdes que mudavam "
        "de tonalidade conforme as nuvens passavam lentamente pelo céu. Em cada parada, "
        "passageiros carregavam malas gastas, cestos de frutas e caixas amarradas com "
        "barbante. O som cadenciado dos trilhos misturava-se às conversas baixas nos "
        "vagões de madeira, enquanto o vento suave entrava pelas frestas e refrescava o "
        "ambiente quente da tarde. Ao longe, a serra azulada anunciava que o destino "
        "estava cada vez mais próximo."
    ),
    (
        "A engenharia de áudio embarcado exige um equilíbrio cuidadoso entre precisão "
        "matemática, robustez acústica e baixo consumo de energia. Pequenos microcontroladores "
        "conseguem capturar sinais digitais em alta frequência, filtrar ruídos estacionários "
        "do ambiente e extrair características espectrais da voz humana em tempo real, sem "
        "depender de conexão com servidores externos. Quando o algoritmo respeita os limites "
        "de memória e processamento da placa, torna-se possível construir sistemas portáteis, "
        "privados e responsivos que operam de forma confiável mesmo em salas com ventiladores "
        "ou conversas distantes ao fundo."
    ),
    (
        "Na feira livre de domingo, as barracas coloridas exibiam frutas tropicais, "
        "hortaliças frescas, queijos artesanais e ramos de flores recém-colhidas. Os "
        "vendedores anunciavam os preços com entusiasmo, oferecendo amostras de manga, "
        "abacaxi e jabuticaba para quem passava devagar pelos corredores movimentados. "
        "Perto da barraca de temperos, o aroma de pimenta, orégano e canela misturava-se "
        "ao cheiro de pastel frito na hora. Famílias inteiras aproveitavam a manhã amena "
        "para escolher ingredientes da semana e conversar calmamente com amigos encontrados "
        "pelo caminho."
    ),
    (
        "Quando a chuva forte chegou de repente no fim da tarde, as luzes dos postes e "
        "das vitrines passaram a refletir intensamente no asfalto molhado das avenidas. "
        "Quem caminhava pela praça correu para se abrigar sob as marquises enquanto o "
        "vento balançava os galhos das árvores mais altas. Dentro da biblioteca silenciosa, "
        "porém, o ritmo permanecia sereno: estudantes revisavam anotações em cadernos "
        "abertos, folheavam capítulos técnicos e organizavam livros empilhados sobre as "
        "grandes mesas de madeira clara, esperando a tempestade diminuir antes de voltar "
        "para casa."
    ),
    (
        "Antes de iniciar qualquer experimento em laboratório, é fundamental conferir todas "
        "as conexões elétricas, anotar as condições do ambiente e verificar se os sensores "
        "estão devidamente calibrados. Pequenos detalhes de montagem, como a distância até "
        "o microfone ou um cabo mal encaixado, podem introduzir interferências difíceis de "
        "corrigir depois na etapa de análise. Registrar cada sessão de forma organizada, com "
        "nomes padronizados e observações claras, economiza horas de retrabalho e garante que "
        "outros pesquisadores consigam reproduzir os resultados com segurança."
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
    for cmd in (
        ["paplay", str(path)],
        ["aplay", "-q", str(path)],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
    ):
        if shutil.which(cmd[0]):
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    return False


# Paleta Gruvbox Dark
GB = {
    "bg0_h": "#1d2021",
    "bg0": "#282828",
    "bg1": "#3c3836",
    "bg2": "#504945",
    "bg3": "#665c54",
    "fg0": "#fbf1c7",
    "fg1": "#ebdbb2",
    "fg4": "#a89984",
    "red": "#cc241d",
    "red_hi": "#fb4934",
    "green": "#98971a",
    "green_hi": "#b8bb26",
    "yellow": "#d79921",
    "yellow_hi": "#fabd2f",
    "blue": "#458588",
    "blue_hi": "#83a598",
    "aqua": "#8ec07c",
    "orange": "#fe8019",
}


class ColetorApp(tk.Tk if tk is not None else object):
    def __init__(self):
        super().__init__()
        self.title("RecVoz — Coletor de Dataset (ESP32-S3)")
        self.geometry("940x820")
        self.minsize(840, 720)
        self.configure(bg=GB["bg0"])

        self.recording = False
        self.last_saved = None
        self.prompt_idx = 0
        self.prompt_font_size = 13

        self.port_var = tk.StringVar()
        self.baud_var = tk.IntVar(value=115200)
        self.dur_var = tk.IntVar(value=18)
        self.group_var = tk.StringVar(value="known")
        self.name_var = tk.StringVar(value="adrian")
        self.session_var = tk.IntVar(value=1)
        self.preview_var = tk.StringVar()
        self.banner_var = tk.StringVar(value="Pronto — posicione o microfone (~20 cm) e clique em GRAVAR (ou F5)")
        self.quality_var = tk.StringVar(value="Nenhuma gravação realizada nesta sessão.")
        self.summary_var = tk.StringVar()
        self.prompt_meta_var = tk.StringVar()

        self._setup_style()
        self._build_ui()
        self.bind("<F5>", lambda _e: self._start_record(overwrite=False))
        self._refresh_ports()
        self._on_identity_change()
        self._refresh_dataset_tree()

    def _setup_style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.option_add("*TCombobox*Listbox.background", GB["bg1"])
        self.option_add("*TCombobox*Listbox.foreground", GB["fg1"])
        self.option_add("*TCombobox*Listbox.selectBackground", GB["bg2"])
        self.option_add("*TCombobox*Listbox.selectForeground", GB["yellow_hi"])

        style.configure(".", background=GB["bg0"], foreground=GB["fg1"], font=("TkDefaultFont", 10))
        style.configure("TFrame", background=GB["bg0"])
        style.configure("Card.TLabelframe", background=GB["bg1"], bordercolor=GB["bg2"], lightcolor=GB["bg2"], darkcolor=GB["bg2"], borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background=GB["bg0"], foreground=GB["yellow_hi"], font=("TkDefaultFont", 10, "bold"))
        style.configure("CardInner.TFrame", background=GB["bg1"])
        style.configure("CardInner.TLabel", background=GB["bg1"], foreground=GB["fg1"])
        style.configure("Muted.TLabel", background=GB["bg1"], foreground=GB["fg4"], font=("TkDefaultFont", 9))
        style.configure("CardInner.TRadiobutton", background=GB["bg1"], foreground=GB["fg1"], indicatorcolor=GB["bg0_h"])
        style.map(
            "CardInner.TRadiobutton",
            background=[("active", GB["bg1"])],
            foreground=[("active", GB["yellow_hi"])],
            indicatorcolor=[("selected", GB["yellow_hi"])],
        )
        style.configure("TButton", background=GB["bg2"], foreground=GB["fg1"], bordercolor=GB["bg3"], padding=(10, 4))
        style.map(
            "TButton",
            background=[("active", GB["bg3"]), ("disabled", GB["bg1"])],
            foreground=[("active", GB["fg0"]), ("disabled", GB["fg4"])],
        )
        style.configure("TEntry", fieldbackground=GB["bg0_h"], foreground=GB["fg0"], insertcolor=GB["yellow_hi"], bordercolor=GB["bg3"])
        style.configure("TSpinbox", fieldbackground=GB["bg0_h"], foreground=GB["fg0"], arrowcolor=GB["fg1"], background=GB["bg2"], bordercolor=GB["bg3"])
        style.configure("TCombobox", fieldbackground=GB["bg0_h"], foreground=GB["fg0"], arrowcolor=GB["fg1"], background=GB["bg2"], bordercolor=GB["bg3"])
        style.map("TCombobox", fieldbackground=[("readonly", GB["bg0_h"])], foreground=[("readonly", GB["fg0"])])
        style.configure("Horizontal.TProgressbar", troughcolor=GB["bg0_h"], background=GB["green_hi"], bordercolor=GB["bg2"], lightcolor=GB["green_hi"], darkcolor=GB["green"])
        style.configure("Vertical.TScrollbar", background=GB["bg2"], troughcolor=GB["bg0_h"], arrowcolor=GB["fg1"], bordercolor=GB["bg1"])
        style.configure("Treeview", background=GB["bg0_h"], fieldbackground=GB["bg0_h"], foreground=GB["fg1"], bordercolor=GB["bg2"], rowheight=24)
        style.map("Treeview", background=[("selected", GB["bg2"])], foreground=[("selected", GB["fg0"])])
        style.configure("Treeview.Heading", font=("TkDefaultFont", 9, "bold"), background=GB["bg2"], foreground=GB["yellow_hi"], relief="flat")
        style.map("Treeview.Heading", background=[("active", GB["bg3"])])

    def _build_ui(self):
        pad = {"padx": 12, "pady": 5}

        # Banner de estado ao vivo (visível de longe para o voluntário)
        self.banner = tk.Label(
            self,
            textvariable=self.banner_var,
            bg=GB["bg1"],
            fg=GB["fg0"],
            font=("TkDefaultFont", 12, "bold"),
            pady=10,
            padx=14,
            anchor=tk.W,
        )
        self.banner.pack(fill=tk.X)

        # 1. Conexão e Destino (compactos em linha)
        top_grid = ttk.Frame(self)
        top_grid.pack(fill=tk.X, **pad)
        top_grid.columnconfigure(0, weight=1)
        top_grid.columnconfigure(1, weight=2)

        conn = ttk.LabelFrame(top_grid, text="1. Conexão ESP32-S3", style="Card.TLabelframe")
        conn.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        c_in = ttk.Frame(conn, style="CardInner.TFrame")
        c_in.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        r1 = ttk.Frame(c_in, style="CardInner.TFrame")
        r1.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(r1, text="Porta:", style="CardInner.TLabel").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(r1, textvariable=self.port_var, width=16, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(r1, text="Atualizar", command=self._refresh_ports).pack(side=tk.LEFT)

        r2 = ttk.Frame(c_in, style="CardInner.TFrame")
        r2.pack(fill=tk.X)
        ttk.Label(r2, text="Duração (s):", style="CardInner.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(r2, from_=3, to=60, increment=3, textvariable=self.dur_var, width=4).pack(side=tk.LEFT, padx=(6, 8))
        for preset in (10, 18, 24, 30):
            ttk.Button(r2, text=f"{preset}s", width=4, command=lambda s=preset: self.dur_var.set(s)).pack(side=tk.LEFT, padx=2)

        meta = ttk.LabelFrame(top_grid, text="2. Destino e Locutor", style="Card.TLabelframe")
        meta.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        m_in = ttk.Frame(meta, style="CardInner.TFrame")
        m_in.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        grp_row = ttk.Frame(m_in, style="CardInner.TFrame")
        grp_row.pack(fill=tk.X, pady=(0, 6))
        for key, label, _ in GROUPS:
            ttk.Radiobutton(
                grp_row,
                text=label,
                value=key,
                variable=self.group_var,
                command=self._on_identity_change,
                style="CardInner.TRadiobutton",
            ).pack(side=tk.LEFT, padx=(0, 10))

        id_row = ttk.Frame(m_in, style="CardInner.TFrame")
        id_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(id_row, text="Nome:", style="CardInner.TLabel").pack(side=tk.LEFT)
        name_entry = ttk.Entry(id_row, textvariable=self.name_var, width=18)
        name_entry.pack(side=tk.LEFT, padx=(6, 12))
        name_entry.bind("<KeyRelease>", lambda _e: self._on_identity_change())

        ttk.Label(id_row, text="Sessão s:", style="CardInner.TLabel").pack(side=tk.LEFT)
        sess_spin = ttk.Spinbox(id_row, from_=1, to=99, textvariable=self.session_var, width=4, command=self._update_preview)
        sess_spin.pack(side=tk.LEFT, padx=(4, 8))
        sess_spin.bind("<KeyRelease>", lambda _e: self._update_preview())
        ttk.Button(id_row, text="Auto sN", command=self._auto_session).pack(side=tk.LEFT)

        ttk.Label(m_in, textvariable=self.preview_var, style="CardInner.TLabel", foreground=GB["aqua"]).pack(anchor=tk.W)

        # 3. Teleprompter (grande, confortável e com textos longos)
        prompt_box = ttk.LabelFrame(
            self,
            text="3. Teleprompter — espere ~1 s em silêncio no início e leia em voz natural",
            style="Card.TLabelframe",
        )
        prompt_box.pack(fill=tk.X, **pad)
        p_in = ttk.Frame(prompt_box, style="CardInner.TFrame")
        p_in.pack(fill=tk.X, padx=10, pady=8)

        txt_wrap = ttk.Frame(p_in, style="CardInner.TFrame")
        txt_wrap.pack(fill=tk.X)
        self.prompt_text = tk.Text(
            txt_wrap,
            height=6,
            wrap=tk.WORD,
            font=("TkDefaultFont", self.prompt_font_size),
            bg=GB["bg0_h"],
            fg=GB["fg0"],
            insertbackground=GB["yellow_hi"],
            selectbackground=GB["bg2"],
            selectforeground=GB["fg0"],
            highlightbackground=GB["bg2"],
            highlightcolor=GB["yellow_hi"],
            highlightthickness=1,
            relief=tk.FLAT,
            padx=14,
            pady=10,
            spacing1=3,
            spacing2=4,
            spacing3=3,
        )
        p_scroll = ttk.Scrollbar(txt_wrap, orient=tk.VERTICAL, command=self.prompt_text.yview)
        self.prompt_text.configure(yscrollcommand=p_scroll.set)
        self.prompt_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        p_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._set_prompt(0)

        p_btns = ttk.Frame(p_in, style="CardInner.TFrame")
        p_btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(p_btns, text="◀ Anterior", command=lambda: self._set_prompt(self.prompt_idx - 1)).pack(side=tk.LEFT)
        ttk.Button(p_btns, text="Próximo texto ▶", command=self._next_prompt).pack(side=tk.LEFT, padx=6)
        ttk.Button(p_btns, text="A-", width=3, command=lambda: self._change_font(-1)).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(p_btns, text="A+", width=3, command=lambda: self._change_font(1)).pack(side=tk.LEFT, padx=2)
        ttk.Label(p_btns, textvariable=self.prompt_meta_var, style="Muted.TLabel").pack(side=tk.RIGHT)

        # 4. Controles de Gravação
        rec_box = ttk.LabelFrame(self, text="4. Captura e Checagem Automática", style="Card.TLabelframe")
        rec_box.pack(fill=tk.X, **pad)
        r_in = ttk.Frame(rec_box, style="CardInner.TFrame")
        r_in.pack(fill=tk.X, padx=10, pady=8)

        btn_row = ttk.Frame(r_in, style="CardInner.TFrame")
        btn_row.pack(fill=tk.X, pady=(0, 6))

        self.rec_btn = tk.Button(
            btn_row,
            text="●  GRAVAR  (F5)",
            command=lambda: self._start_record(overwrite=False),
            bg=GB["red"],
            fg=GB["fg0"],
            activebackground=GB["red_hi"],
            activeforeground=GB["fg0"],
            disabledforeground=GB["fg4"],
            font=("TkDefaultFont", 12, "bold"),
            relief=tk.FLAT,
            padx=22,
            pady=7,
            cursor="hand2",
        )
        self.rec_btn.pack(side=tk.LEFT)

        self.rerec_btn = ttk.Button(
            btn_row, text="Regravar último", command=lambda: self._start_record(overwrite=True), state=tk.DISABLED
        )
        self.rerec_btn.pack(side=tk.LEFT, padx=8)

        self.play_btn = ttk.Button(btn_row, text="▶ Ouvir último", command=self._play_last, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=4)

        self.prog = ttk.Progressbar(btn_row, mode="determinate", length=280)
        self.prog.pack(side=tk.RIGHT, padx=4)

        self.quality_lbl = ttk.Label(r_in, textvariable=self.quality_var, style="CardInner.TLabel", font=("TkDefaultFont", 10, "bold"))
        self.quality_lbl.pack(anchor=tk.W)

        # 5. Progresso do Dataset
        ds_box = ttk.LabelFrame(self, text="5. Dataset Atual no Disco", style="Card.TLabelframe")
        ds_box.pack(fill=tk.BOTH, expand=True, **pad)
        d_in = ttk.Frame(ds_box, style="CardInner.TFrame")
        d_in.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        top_ds = ttk.Frame(d_in, style="CardInner.TFrame")
        top_ds.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(top_ds, textvariable=self.summary_var, style="CardInner.TLabel", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        ttk.Button(top_ds, text="Recarregar lista", command=self._refresh_dataset_tree).pack(side=tk.RIGHT)

        tree_wrap = ttk.Frame(d_in, style="CardInner.TFrame")
        tree_wrap.pack(fill=tk.BOTH, expand=True)
        cols = ("grupo", "arquivo", "dur", "janelas", "ruido")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=7)
        self.tree.heading("grupo", text="Grupo")
        self.tree.heading("arquivo", text="Arquivo")
        self.tree.heading("dur", text="Duração")
        self.tree.heading("janelas", text="Janelas VAD")
        self.tree.heading("ruido", text="Auto-Ruído")
        self.tree.column("grupo", width=155, anchor=tk.W)
        self.tree.column("arquivo", width=360, anchor=tk.W)
        self.tree.column("dur", width=85, anchor=tk.CENTER)
        self.tree.column("janelas", width=95, anchor=tk.CENTER)
        self.tree.column("ruido", width=95, anchor=tk.CENTER)
        self.tree.tag_configure("ok", foreground=GB["green_hi"])
        self.tree.tag_configure("warn", foreground=GB["orange"])

        t_scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=t_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        t_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _set_banner(self, text, bg="#e2e8f0", fg="#0f172a"):
        self.banner_var.set(text)
        self.banner.configure(bg=bg, fg=fg)

    def _change_font(self, delta):
        self.prompt_font_size = max(10, min(22, self.prompt_font_size + delta))
        self.prompt_text.configure(font=("TkDefaultFont", self.prompt_font_size))

    def _refresh_ports(self):
        ports = find_ports()
        self.port_combo["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        elif not ports:
            self.port_var.set("")

    def _set_prompt(self, idx):
        self.prompt_idx = idx % len(PROMPTS)
        txt = PROMPTS[self.prompt_idx]
        words = len(txt.split())
        est_sec = round(words / 2.3)
        self.prompt_text.config(state=tk.NORMAL)
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", txt)
        self.prompt_text.config(state=tk.DISABLED)
        self.prompt_meta_var.set(f"Texto {self.prompt_idx + 1}/{len(PROMPTS)}  •  {words} palavras (~{est_sec} s de leitura)")

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
            self.preview_var.set("Saída: (informe um nome válido)")
        else:
            rel = out.relative_to(ROOT)
            exists_tag = "  [JÁ EXISTE]" if out.exists() else ""
            self.preview_var.set(f"Saída: {rel}{exists_tag}")

    def _set_busy(self, busy):
        self.recording = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.rec_btn.config(state=state, text="GRAVANDO..." if busy else "●  GRAVAR  (F5)")
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
        self._set_banner(f"Conectando em {port}... prepare-se (1 s de silêncio no início)", bg=GB["yellow"], fg=GB["bg0_h"])
        self.quality_var.set("Capturando áudio do ESP32-S3...")
        self.quality_lbl.config(foreground=GB["fg4"])

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
        if idx == 1:
            self._set_banner(
                f"● GRAVANDO [{idx}/{total}] ({sec}s) — 1 s de silêncio e comece a ler o texto!",
                bg=GB["red"],
                fg=GB["fg0"],
            )
        else:
            self._set_banner(
                f"● GRAVANDO [{idx}/{total}] ({sec}s) — CONTINUE LENDO NORMALMENTE",
                bg=GB["red"],
                fg=GB["fg0"],
            )

    def _on_record_done(self, out_path, stats, group_key, overwrite):
        self.prog["value"] = self.prog["maximum"]
        self.last_saved = out_path
        self._set_busy(False)

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
            f"{stats['duration']:.1f}s  |  {stats['windows']} janelas válidas  |  "
            f"pico={stats['peak']:.2f}  |  auto-ruído: {noise_txt}"
        )
        rel = out_path.relative_to(ROOT)
        if warns:
            self._set_banner(f"⚠ Salvo com ressalvas: {rel}", bg=GB["yellow"], fg=GB["bg0_h"])
            self.quality_var.set(f"⚠ {base_msg} — {', '.join(warns)}")
            self.quality_lbl.config(foreground=GB["orange"])
        else:
            self._set_banner(f"✓ Gravação OK: {rel}", bg=GB["green"], fg=GB["fg0"])
            self.quality_var.set(f"✓ {base_msg}")
            self.quality_lbl.config(foreground=GB["green_hi"])

        if not overwrite:
            self._auto_session()
            self._next_prompt()
        self._update_preview()
        self._refresh_dataset_tree()

    def _on_record_error(self, exc):
        self._set_busy(False)
        self._set_banner("✖ Erro na gravação — veja os detalhes abaixo", bg=GB["red"], fg=GB["fg0"])
        self.quality_var.set(str(exc))
        self.quality_lbl.config(foreground=GB["red_hi"])
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
                tag = "ok"
                try:
                    st = analyze_wav(p)
                    dur_s = f"{st['duration']:.1f}s"
                    win_s = str(st["windows"])
                    ns_s = "OK" if st["has_auto_noise"] else "—"
                    if key != "noise" and (st["windows"] < 5 or not st["has_auto_noise"]):
                        tag = "warn"
                except Exception:
                    dur_s, win_s, ns_s, tag = "?", "?", "?", "warn"
                self.tree.insert("", tk.END, values=(label, p.name, dur_s, win_s, ns_s), tags=(tag,))
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
            f"Conhecidos (>=3 sessões): {ready_known}/{len(known_sessions)}   |   "
            f"Impostores Cal: {len(counts['cal'])} (meta 2+)   |   "
            f"Impostores Teste: {len(counts['test'])} (meta 3+)   |   "
            f"Ruído: {counts['noise']}"
        )


def main():
    if tk is None:
        raise SystemExit("Tkinter não está instalado neste Python (instale python3-tk).")
    app = ColetorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
