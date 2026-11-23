#!/usr/bin/env python3
"""Interface moderna (Gruvbox Dark) para coletar o dataset RecVoz direto do ESP32-S3."""
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
    from tkinter import font as tkfont
    from tkinter import messagebox, ttk
except ImportError:  # ambiente headless/CI sem python3-tk
    tk = None
    tkfont = None
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
    ("cal", "Impostor Cal", ROOT / "dados" / "desconhecidos_cal"),
    ("test", "Impostor Teste", ROOT / "dados" / "desconhecidos_teste"),
    ("noise", "Ruído", ROOT / "dados" / "ruido"),
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

# Paleta Gruvbox Dark com elevação por superfície (sem bordas pesadas)
GB = {
    "bg0_h": "#1d2021",
    "bg0": "#282828",
    "bg_side": "#32302f",
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
    "aqua": "#8ec07c",
    "orange": "#fe8019",
    "blue_hi": "#83a598",
}


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


def pick_font_family(root):
    if tkfont is None:
        return "TkDefaultFont"
    available = set(tkfont.families(root))
    for candidate in ("Inter", "Ubuntu", "Cantarell", "Noto Sans", "DejaVu Sans", "Liberation Sans"):
        if candidate in available:
            return candidate
    return "TkDefaultFont"


class ColetorApp(tk.Tk if tk is not None else object):
    def __init__(self):
        super().__init__()
        self.title("RecVoz — Coletor")
        self.geometry("980x680")
        self.minsize(880, 600)
        self.configure(bg=GB["bg0"])

        self.ui_font = pick_font_family(self)
        self.recording = False
        self.last_saved = None
        self.prompt_idx = 0
        self.prompt_font_size = 12
        self.tree_paths = {}

        self.port_var = tk.StringVar()
        self.baud_var = tk.IntVar(value=115200)
        self.dur_var = tk.IntVar(value=18)
        self.group_var = tk.StringVar(value="known")
        self.name_var = tk.StringVar(value="adrian")
        self.session_var = tk.IntVar(value=1)
        self.preview_var = tk.StringVar()
        self.status_badge_var = tk.StringVar(value="PRONTO")
        self.status_hint_var = tk.StringVar(value="1 s de silêncio e leia o texto")
        self.quality_var = tk.StringVar(value="Nenhuma gravação nesta sessão")
        self.prompt_meta_var = tk.StringVar()

        self.group_btns = {}
        self.dur_btns = {}
        self.metric_vars = {
            "known": tk.StringVar(value="Conhecidos (≥3): 0/0"),
            "cal": tk.StringVar(value="Imp. Cal: 0/2+"),
            "test": tk.StringVar(value="Imp. Teste: 0/3+"),
        }

        self._setup_style()
        self._build_ui()
        self.bind("<F5>", lambda _e: self._start_record(overwrite=False))
        self._refresh_ports()
        self._select_group("known")
        self._select_duration(18)
        self._refresh_dataset_tree()

    def _f(self, size=10, weight="normal"):
        return (self.ui_font, size, weight)

    def _setup_style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.option_add("*TCombobox*Listbox.background", GB["bg1"])
        self.option_add("*TCombobox*Listbox.foreground", GB["fg0"])
        self.option_add("*TCombobox*Listbox.selectBackground", GB["bg2"])
        self.option_add("*TCombobox*Listbox.selectForeground", GB["yellow_hi"])
        self.option_add("*TCombobox*Listbox.font", self._f(9))

        style.configure("TCombobox", fieldbackground=GB["bg0_h"], foreground=GB["fg0"], arrowcolor=GB["fg1"], background=GB["bg1"], bordercolor=GB["bg2"], padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", GB["bg0_h"])], foreground=[("readonly", GB["fg0"])])
        style.configure("Horizontal.TProgressbar", troughcolor=GB["bg0_h"], background=GB["green_hi"], bordercolor=GB["bg0_h"], lightcolor=GB["green_hi"], darkcolor=GB["green"], thickness=6)
        style.configure("Vertical.TScrollbar", background=GB["bg2"], troughcolor=GB["bg0_h"], arrowcolor=GB["fg4"], bordercolor=GB["bg0_h"], relief="flat")
        style.configure("Treeview", background=GB["bg0_h"], fieldbackground=GB["bg0_h"], foreground=GB["fg1"], bordercolor=GB["bg0_h"], borderwidth=0, rowheight=26, font=self._f(9))
        style.map("Treeview", background=[("selected", GB["bg2"])], foreground=[("selected", GB["fg0"])])
        style.configure("Treeview.Heading", font=self._f(9, "bold"), background=GB["bg1"], foreground=GB["fg4"], relief="flat", padding=(8, 5))
        style.map("Treeview.Heading", background=[("active", GB["bg2"])])

    def _flat_btn(self, parent, text, command, bg=None, fg=None, hover=None, font=None, padx=12, pady=6, width=None):
        bg = bg or GB["bg1"]
        fg = fg or GB["fg1"]
        hover = hover or GB["bg2"]
        kw = dict(
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=GB["fg0"],
            disabledforeground=GB["bg3"],
            font=font or self._f(10),
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            padx=padx,
            pady=pady,
            cursor="hand2",
        )
        if width is not None:
            kw["width"] = width
        return tk.Button(parent, **kw)

    def _build_ui(self):
        # ── TOP BAR ──────────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=GB["bg0_h"], height=50)
        topbar.pack(fill=tk.X)
        topbar.pack_propagate(False)

        brand = tk.Frame(topbar, bg=GB["bg0_h"])
        brand.pack(side=tk.LEFT, padx=18)
        tk.Label(brand, text="●", bg=GB["bg0_h"], fg=GB["yellow_hi"], font=self._f(12, "bold")).pack(side=tk.LEFT)
        tk.Label(brand, text=" RecVoz", bg=GB["bg0_h"], fg=GB["fg0"], font=self._f(12, "bold")).pack(side=tk.LEFT)
        tk.Label(brand, text="  Coletor ESP32-S3", bg=GB["bg0_h"], fg=GB["fg4"], font=self._f(9)).pack(side=tk.LEFT)

        right_bar = tk.Frame(topbar, bg=GB["bg0_h"])
        right_bar.pack(side=tk.RIGHT, padx=18)

        tk.Label(right_bar, text="Duração", bg=GB["bg0_h"], fg=GB["fg4"], font=self._f(9)).pack(side=tk.LEFT, padx=(0, 6))
        for sec in (10, 18, 24, 30):
            b = self._flat_btn(right_bar, f"{sec}s", lambda s=sec: self._select_duration(s), padx=8, pady=3, font=self._f(9, "bold"))
            b.pack(side=tk.LEFT, padx=2)
            self.dur_btns[sec] = b

        tk.Frame(right_bar, bg=GB["bg2"], width=1, height=20).pack(side=tk.LEFT, padx=12)

        tk.Label(right_bar, text="Porta", bg=GB["bg0_h"], fg=GB["fg4"], font=self._f(9)).pack(side=tk.LEFT, padx=(0, 6))
        self.port_combo = ttk.Combobox(right_bar, textvariable=self.port_var, width=14, state="readonly", font=self._f(9))
        self.port_combo.pack(side=tk.LEFT)
        self._flat_btn(right_bar, "↻", self._refresh_ports, padx=8, pady=3, font=self._f(9, "bold")).pack(side=tk.LEFT, padx=(4, 0))

        main = tk.Frame(self, bg=GB["bg0"])
        main.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)

        # ── FAIXA 1: CONFIGURAÇÃO RÁPIDA (1 CARTÃO COMPACTO) ────────────────
        cfg = tk.Frame(main, bg=GB["bg_side"], padx=16, pady=12)
        cfg.pack(fill=tk.X, pady=(0, 12))

        cfg_top = tk.Frame(cfg, bg=GB["bg_side"])
        cfg_top.pack(fill=tk.X)

        left_cfg = tk.Frame(cfg_top, bg=GB["bg_side"])
        left_cfg.pack(side=tk.LEFT)
        tk.Label(left_cfg, text="TIPO", bg=GB["bg_side"], fg=GB["fg4"], font=self._f(8, "bold")).pack(anchor=tk.W, pady=(0, 5))
        grp_row = tk.Frame(left_cfg, bg=GB["bg_side"])
        grp_row.pack(anchor=tk.W)
        for key, label, _ in GROUPS:
            b = self._flat_btn(grp_row, label, lambda k=key: self._select_group(k), padx=11, pady=6, font=self._f(9, "bold"))
            b.pack(side=tk.LEFT, padx=(0, 6))
            self.group_btns[key] = b

        right_cfg = tk.Frame(cfg_top, bg=GB["bg_side"])
        right_cfg.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(18, 0))
        hdr_id = tk.Frame(right_cfg, bg=GB["bg_side"])
        hdr_id.pack(fill=tk.X, pady=(0, 5))
        tk.Label(hdr_id, text="LOCUTOR E SESSÃO", bg=GB["bg_side"], fg=GB["fg4"], font=self._f(8, "bold")).pack(side=tk.LEFT)
        tk.Label(hdr_id, textvariable=self.preview_var, bg=GB["bg_side"], fg=GB["aqua"], font=self._f(9)).pack(side=tk.RIGHT)

        id_row = tk.Frame(right_cfg, bg=GB["bg_side"])
        id_row.pack(fill=tk.X)
        self.name_entry = tk.Entry(
            id_row,
            textvariable=self.name_var,
            bg=GB["bg0_h"],
            fg=GB["fg0"],
            insertbackground=GB["yellow_hi"],
            relief=tk.FLAT,
            font=self._f(10, "bold"),
            highlightthickness=1,
            highlightbackground=GB["bg2"],
            highlightcolor=GB["yellow_hi"],
        )
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, ipadx=8)
        self.name_entry.bind("<KeyRelease>", lambda _e: self._on_identity_change())

        sess_box = tk.Frame(id_row, bg=GB["bg0_h"], highlightthickness=1, highlightbackground=GB["bg2"])
        sess_box.pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(sess_box, text=" s", bg=GB["bg0_h"], fg=GB["fg4"], font=self._f(10, "bold")).pack(side=tk.LEFT)
        self.sess_entry = tk.Entry(
            sess_box,
            textvariable=self.session_var,
            width=3,
            bg=GB["bg0_h"],
            fg=GB["yellow_hi"],
            insertbackground=GB["yellow_hi"],
            relief=tk.FLAT,
            font=self._f(10, "bold"),
            justify=tk.CENTER,
        )
        self.sess_entry.pack(side=tk.LEFT, ipady=5)
        self.sess_entry.bind("<KeyRelease>", lambda _e: self._update_preview())

        # ── FAIXA 2: TELEPROMPTER COMPACTO (ESQ) + CONTROLES (DIR) ──────────
        mid = tk.Frame(main, bg=GB["bg0"])
        mid.pack(fill=tk.X, pady=(0, 12))
        mid.columnconfigure(0, weight=13, uniform="mid")
        mid.columnconfigure(1, weight=8, uniform="mid")

        # Esquerda: Teleprompter compacto
        tp_card = tk.Frame(mid, bg=GB["bg_side"], padx=14, pady=12)
        tp_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tp_hdr = tk.Frame(tp_card, bg=GB["bg_side"])
        tp_hdr.pack(fill=tk.X, pady=(0, 8))
        tk.Label(tp_hdr, text="TEXTO PARA LEITURA", bg=GB["bg_side"], fg=GB["fg4"], font=self._f(8, "bold")).pack(side=tk.LEFT)
        tk.Label(tp_hdr, textvariable=self.prompt_meta_var, bg=GB["bg_side"], fg=GB["fg4"], font=self._f(8)).pack(side=tk.LEFT, padx=(10, 0))

        self._flat_btn(tp_hdr, "▶", self._next_prompt, padx=7, pady=2, font=self._f(8, "bold")).pack(side=tk.RIGHT)
        self._flat_btn(tp_hdr, "◀", lambda: self._set_prompt(self.prompt_idx - 1), padx=7, pady=2, font=self._f(8, "bold")).pack(side=tk.RIGHT, padx=(0, 3))
        self._flat_btn(tp_hdr, "A+", lambda: self._change_font(1), padx=6, pady=2, font=self._f(8, "bold")).pack(side=tk.RIGHT, padx=(8, 3))
        self._flat_btn(tp_hdr, "A-", lambda: self._change_font(-1), padx=6, pady=2, font=self._f(8, "bold")).pack(side=tk.RIGHT, padx=(0, 2))

        tp_wrap = tk.Frame(tp_card, bg=GB["bg0_h"])
        tp_wrap.pack(fill=tk.BOTH, expand=True)
        self.prompt_text = tk.Text(
            tp_wrap,
            height=6,
            wrap=tk.WORD,
            font=self._f(self.prompt_font_size),
            bg=GB["bg0_h"],
            fg=GB["fg0"],
            selectbackground=GB["bg2"],
            selectforeground=GB["fg0"],
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=10,
            spacing1=2,
            spacing2=5,
            spacing3=2,
        )
        p_scroll = ttk.Scrollbar(tp_wrap, orient=tk.VERTICAL, command=self.prompt_text.yview)
        self.prompt_text.configure(yscrollcommand=p_scroll.set)
        self.prompt_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        p_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._set_prompt(0)

        # Direita: Controles de gravação
        ctrl = tk.Frame(mid, bg=GB["bg_side"], padx=16, pady=12)
        ctrl.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        st_row = tk.Frame(ctrl, bg=GB["bg_side"])
        st_row.pack(fill=tk.X, pady=(0, 8))
        self.badge_lbl = tk.Label(st_row, textvariable=self.status_badge_var, bg=GB["bg2"], fg=GB["fg0"], font=self._f(8, "bold"), padx=9, pady=3)
        self.badge_lbl.pack(side=tk.LEFT)
        self.hint_lbl = tk.Label(st_row, textvariable=self.status_hint_var, bg=GB["bg_side"], fg=GB["fg1"], font=self._f(9), padx=8)
        self.hint_lbl.pack(side=tk.LEFT, fill=tk.X)

        self.rec_btn = self._flat_btn(
            ctrl,
            "●   GRAVAR   (F5)",
            lambda: self._start_record(overwrite=False),
            bg=GB["red"],
            fg=GB["fg0"],
            hover=GB["red_hi"],
            font=self._f(12, "bold"),
            padx=16,
            pady=11,
        )
        self.rec_btn.pack(fill=tk.X, pady=(2, 10))

        self.prog = ttk.Progressbar(ctrl, mode="determinate")
        self.prog.pack(fill=tk.X, pady=(0, 10))

        sec_row = tk.Frame(ctrl, bg=GB["bg_side"])
        sec_row.pack(fill=tk.X)
        sec_row.columnconfigure(0, weight=1)
        sec_row.columnconfigure(1, weight=1)

        self.play_btn = self._flat_btn(sec_row, "▶  Ouvir último", self._play_last, padx=10, pady=7, font=self._f(9, "bold"))
        self.play_btn.configure(state=tk.DISABLED)
        self.play_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.rerec_btn = self._flat_btn(sec_row, "↻  Regravar", lambda: self._start_record(overwrite=True), padx=10, pady=7, font=self._f(9, "bold"))
        self.rerec_btn.configure(state=tk.DISABLED)
        self.rerec_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.quality_lbl = tk.Label(ctrl, textvariable=self.quality_var, bg=GB["bg_side"], fg=GB["fg4"], font=self._f(9, "bold"), anchor=tk.W, wraplength=310, justify=tk.LEFT)
        self.quality_lbl.pack(fill=tk.X, pady=(10, 0))

        # ── FAIXA 3: DATASET E PROGRESSO (LARGURA TOTAL) ────────────────────
        ds_card = tk.Frame(main, bg=GB["bg_side"], padx=14, pady=12)
        ds_card.pack(fill=tk.BOTH, expand=True)

        ds_hdr = tk.Frame(ds_card, bg=GB["bg_side"])
        ds_hdr.pack(fill=tk.X, pady=(0, 8))
        tk.Label(ds_hdr, text="DATASET GRAVADO", bg=GB["bg_side"], fg=GB["fg4"], font=self._f(8, "bold")).pack(side=tk.LEFT)
        tk.Label(ds_hdr, text="(duplo clique para ouvir)", bg=GB["bg_side"], fg=GB["bg3"], font=self._f(8)).pack(side=tk.LEFT, padx=(8, 0))

        self._flat_btn(ds_hdr, "↻", self._refresh_dataset_tree, padx=8, pady=2, font=self._f(9, "bold")).pack(side=tk.RIGHT)
        for k in ("test", "cal", "known"):
            pill = tk.Label(ds_hdr, textvariable=self.metric_vars[k], bg=GB["bg1"], fg=GB["yellow_hi"], font=self._f(8, "bold"), padx=9, pady=3)
            pill.pack(side=tk.RIGHT, padx=(0, 6))

        tree_wrap = tk.Frame(ds_card, bg=GB["bg0_h"])
        tree_wrap.pack(fill=tk.BOTH, expand=True)
        cols = ("grupo", "arquivo", "dur", "janelas", "ruido")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings")
        self.tree.heading("grupo", text="Grupo")
        self.tree.heading("arquivo", text="Arquivo")
        self.tree.heading("dur", text="Duração")
        self.tree.heading("janelas", text="Janelas VAD")
        self.tree.heading("ruido", text="Auto-Ruído")
        self.tree.column("grupo", width=150, anchor=tk.W)
        self.tree.column("arquivo", width=360, anchor=tk.W)
        self.tree.column("dur", width=90, anchor=tk.CENTER)
        self.tree.column("janelas", width=105, anchor=tk.CENTER)
        self.tree.column("ruido", width=105, anchor=tk.CENTER)
        self.tree.tag_configure("ok", foreground=GB["green_hi"])
        self.tree.tag_configure("warn", foreground=GB["orange"])
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        t_scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=t_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        t_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ── INTERAÇÃO ────────────────────────────────────────────────────────────
    def _select_group(self, key):
        self.group_var.set(key)
        for k, btn in self.group_btns.items():
            if k == key:
                btn.configure(bg=GB["yellow_hi"], fg=GB["bg0_h"], activebackground=GB["yellow"])
            else:
                btn.configure(bg=GB["bg1"], fg=GB["fg1"], activebackground=GB["bg2"])
        self._on_identity_change()

    def _select_duration(self, sec):
        self.dur_var.set(sec)
        for s, btn in self.dur_btns.items():
            if s == sec:
                btn.configure(bg=GB["yellow_hi"], fg=GB["bg0_h"], activebackground=GB["yellow"])
            else:
                btn.configure(bg=GB["bg1"], fg=GB["fg1"], activebackground=GB["bg2"])

    def _set_status(self, badge, hint, badge_bg, badge_fg=None):
        self.status_badge_var.set(badge)
        self.status_hint_var.set(hint)
        self.badge_lbl.configure(bg=badge_bg, fg=badge_fg or GB["fg0"])

    def _change_font(self, delta):
        self.prompt_font_size = max(11, min(24, self.prompt_font_size + delta))
        self.prompt_text.configure(font=self._f(self.prompt_font_size))

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
        self.prompt_meta_var.set(f"Texto {self.prompt_idx + 1}/{len(PROMPTS)}  ·  {words} palavras (~{est_sec}s)")

    def _next_prompt(self):
        self._set_prompt(self.prompt_idx + 1)

    def _auto_session(self):
        group_key = self.group_var.get()
        folder = dict((k, d) for k, _, d in GROUPS)[group_key]
        name = sanitize_name(self.name_var.get())
        if group_key == "noise":
            existing = sorted(folder.glob(f"{name}_s*.wav")) if (folder.exists() and name) else []
            self.session_var.set(len(existing) + 1)
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
            self.preview_var.set("Digite um nome para o locutor")
        else:
            rel = out.relative_to(ROOT)
            exists_tag = "  (já existe)" if out.exists() else ""
            self.preview_var.set(f"→ {rel}{exists_tag}")

    def _set_busy(self, busy):
        self.recording = busy
        self.rec_btn.config(
            state=tk.DISABLED if busy else tk.NORMAL,
            text="●   GRAVANDO..." if busy else "●   GRAVAR   (F5)",
            bg=GB["bg2"] if busy else GB["red"],
        )
        sec_state = tk.DISABLED if (busy or not self.last_saved) else tk.NORMAL
        self.rerec_btn.config(state=sec_state)
        self.play_btn.config(state=sec_state)

    def _start_record(self, overwrite=False):
        if self.recording:
            return
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Porta não encontrada", "Conecte o ESP32-S3 e clique em ↻ ao lado da porta.")
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
                messagebox.showerror("Nome inválido", "Informe um nome de locutor válido.")
                return
            if out_path.exists():
                if not messagebox.askyesno("Sobrescrever?", f"{out_path.name} já existe. Sobrescrever?"):
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
        self._set_status("CONECTANDO", f"Abrindo {port}... prepare-se (1 s de silêncio no início)", GB["yellow_hi"], GB["bg0_h"])
        self.quality_var.set("")

        threading.Thread(
            target=self._record_worker,
            args=(port, self.baud_var.get(), dur, out_path, self.group_var.get(), overwrite),
            daemon=True,
        ).start()

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
            self._set_status(f"● GRAVANDO {idx}/{total}", "1 segundo de silêncio e comece a ler o texto!", GB["red"], GB["fg0"])
        else:
            self._set_status(f"● GRAVANDO {idx}/{total}", f"Continue lendo normalmente ({sec}s)...", GB["red"], GB["fg0"])

    def _on_record_done(self, out_path, stats, group_key, overwrite):
        self.prog["value"] = self.prog["maximum"]
        self.last_saved = out_path
        self._set_busy(False)

        warns = []
        if group_key != "noise":
            if stats["windows"] < 5:
                warns.append(f"poucas janelas ({stats['windows']}; ideal ≥ 5)")
            if not stats["has_auto_noise"]:
                warns.append("sem pausa p/ auto-ruído")
        if stats["peak"] >= 0.98:
            warns.append("clipping")

        noise_txt = "auto-ruído OK" if stats["has_auto_noise"] else "sem pausa"
        summary = f"{stats['duration']:.1f}s  ·  {stats['windows']} janelas válidas  ·  pico {stats['peak']:.2f}  ·  {noise_txt}"
        rel = out_path.relative_to(ROOT)

        if warns:
            self._set_status("⚠ REVISAR", f"Salvo em {rel} ({', '.join(warns)})", GB["orange"], GB["bg0_h"])
            self.quality_var.set(f"⚠  {summary}")
            self.quality_lbl.config(fg=GB["orange"])
        else:
            self._set_status("✓ SALVO OK", str(rel), GB["green"], GB["fg0"])
            self.quality_var.set(f"✓  {summary}")
            self.quality_lbl.config(fg=GB["green_hi"])

        if not overwrite:
            self._auto_session()
            self._next_prompt()
        self._update_preview()
        self._refresh_dataset_tree()

    def _on_record_error(self, exc):
        self._set_busy(False)
        self._set_status("✖ ERRO", "Falha na comunicação com o ESP32-S3", GB["red"], GB["fg0"])
        self.quality_var.set(str(exc))
        self.quality_lbl.config(fg=GB["red_hi"])
        messagebox.showerror("Erro ao gravar", str(exc))

    def _play_last(self):
        if self.last_saved and self.last_saved.exists():
            if not play_wav(self.last_saved):
                messagebox.showwarning("Player indisponível", "Instale aplay, paplay ou ffplay para reproduzir.")

    def _on_tree_double_click(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        path = self.tree_paths.get(sel[0])
        if path and path.exists():
            play_wav(path)

    def _refresh_dataset_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree_paths.clear()

        counts = {"known": set(), "cal": set(), "test": set()}
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
                iid = self.tree.insert("", tk.END, values=(label, p.name, dur_s, win_s, ns_s), tags=(tag,))
                self.tree_paths[iid] = p
                m = re.match(r"^locutor_([a-zA-Z0-9]+)_s\d+\.wav$", p.name)
                if m and key in counts:
                    spk = m.group(1).lower()
                    counts[key].add(spk)
                    if key == "known":
                        known_sessions[spk] = known_sessions.get(spk, 0) + 1

        ready_known = sum(1 for n in known_sessions.values() if n >= 3)
        self.metric_vars["known"].set(f"Conhecidos (≥3): {ready_known}/{len(known_sessions)}")
        self.metric_vars["cal"].set(f"Imp. Cal: {len(counts['cal'])}/2+")
        self.metric_vars["test"].set(f"Imp. Teste: {len(counts['test'])}/3+")


def main():
    if tk is None:
        raise SystemExit("Tkinter não está instalado neste Python (instale python3-tk).")
    app = ColetorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
