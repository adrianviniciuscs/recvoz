#!/usr/bin/env python3
"""Coleta sessões WAV direto do firmware recvoz-esp via comando serial `grab`.

Como `grab` aceita 1..3 s por chamada (limite de RAM da placa), este script
encadeia blocos consecutivos na mesma conexão serial e salva um único WAV
PCM16 mono a 16 kHz.
"""
import argparse
import base64
import glob
import sys
import time
import wave
from pathlib import Path

TARGET_SR = 16000
MAX_CHUNK_SEC = 3


def find_default_port():
    ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
    return ports[0] if ports else None


def plan_chunks(seconds):
    total = int(round(seconds))
    if total < 1 or total > 600:
        raise ValueError('Duração deve ser um inteiro entre 1 e 600 segundos')
    chunks = []
    remaining = total
    while remaining > 0:
        take = min(MAX_CHUNK_SEC, remaining)
        chunks.append(take)
        remaining -= take
    return chunks


def grab_chunk(stream, seconds, timeout=30.0):
    cmd = f'grab {seconds}\n'.encode('ascii')
    stream.write(cmd)
    if hasattr(stream, 'flush'):
        stream.flush()

    deadline = time.monotonic() + timeout
    retry_at = time.monotonic() + 4.0
    started = False
    b64_lines = []
    seen = []

    while time.monotonic() < deadline:
        raw = stream.readline().decode('ascii', errors='ignore').strip()
        if not started and time.monotonic() > retry_at:
            stream.write(cmd)
            if hasattr(stream, 'flush'):
                stream.flush()
            retry_at = time.monotonic() + 4.0
        if not raw:
            continue
        seen.append(raw)
        if len(seen) > 12:
            seen.pop(0)
        if 'Unrecognized command' in raw:
            raise RuntimeError(
                'O firmware na placa não possui o comando `grab`. '
                'Compile e grave `hardware/recvoz-esp` primeiro.'
            )
        if raw.startswith('RECVOZ_B64_START'):
            started = True
            continue
        if raw == 'RECVOZ_B64_END':
            break
        if started:
            b64_lines.append(raw)

    if not started or not b64_lines:
        tail = '\n'.join(seen[-6:]) if seen else '(nenhum dado recebido)'
        raise TimeoutError(f'ESP32 não respondeu ao `grab {seconds}`.\nÚltimas linhas:\n{tail}')

    pcm_bytes = base64.b64decode(''.join(b64_lines))
    if len(pcm_bytes) % 2 != 0:
        raise ValueError(f'PCM truncado: {len(pcm_bytes)} bytes')
    return pcm_bytes


def record_chunks(stream, output, seconds, on_chunk=None):
    chunks = plan_chunks(seconds)
    total_samples = 0
    with wave.open(str(output), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TARGET_SR)
        for idx, sec in enumerate(chunks, start=1):
            if on_chunk is not None:
                on_chunk(idx, len(chunks), sec)
            pcm_bytes = grab_chunk(stream, sec)
            wav.writeframesraw(pcm_bytes)
            total_samples += len(pcm_bytes) // 2
    return total_samples


def open_serial(port, baud):
    import serial

    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.5
    ser.dtr = False
    ser.rts = False
    ser.open()

    boot_deadline = time.monotonic() + 3.0
    while time.monotonic() < boot_deadline:
        raw = ser.readline().decode('ascii', errors='ignore').strip()
        if 'recvoz pronto' in raw or 'recvoz>' in raw:
            break
    time.sleep(0.15)
    ser.reset_input_buffer()
    return ser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--port', help='Porta serial (ex.: /dev/ttyACM0). Padrão: autodetecta.')
    parser.add_argument('-b', '--baud', type=int, default=115200)
    parser.add_argument('-d', '--duration', type=int, default=18, help='Duração total em segundos (padrão: 18).')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Arquivo WAV de saída.')
    parser.add_argument('--force', action='store_true', help='Sobrescreve o arquivo de saída se já existir.')
    args = parser.parse_args()

    try:
        plan_chunks(args.duration)
    except ValueError as exc:
        parser.error(str(exc))

    if args.output.exists() and not args.force:
        parser.error('O arquivo de saída já existe; escolha outro nome ou passe --force')

    port = args.port or find_default_port()
    if not port:
        parser.exit(1, 'Nenhuma porta /dev/ttyACM* ou /dev/ttyUSB* encontrada. Informe -p PORTA.\n')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.wav')

    def progress(idx, total, sec):
        print(f'[{idx}/{total}] Gravando {sec}s via `{port}`...', flush=True)

    try:
        ser = open_serial(port, args.baud)
        try:
            n = record_chunks(ser, partial, args.duration, on_chunk=progress)
        finally:
            ser.close()
        partial.replace(args.output)
        print(f'Salvo: {args.output} ({n} amostras, {n / TARGET_SR:.2f} s)')
    except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
        parser.exit(1, f'Erro: {exc}\nCaptura parcial (se houver): {partial}\n')


if __name__ == '__main__':
    main()
