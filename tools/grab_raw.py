#!/usr/bin/env python3
"""Coleta do firmware externo inmp441_mic: AA55 | n:u16 LE | PCM:i16 LE.

Não compatível com o console do firmware de reconhecimento recvoz.
"""
import argparse
import struct
import time
import wave
from pathlib import Path


def record(stream, output, seconds, timeout=10):
    target = round(seconds * 16000)
    count = 0
    deadline = time.monotonic() + timeout
    with wave.open(str(output), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        while count < target:
            if time.monotonic() > deadline:
                raise TimeoutError(f'Sem frames válidos; recebidas {count} amostras')
            if not stream.read_until(b'\xaa\x55').endswith(b'\xaa\x55'):
                continue
            header = stream.read(2)
            if len(header) != 2:
                continue
            size = struct.unpack('<H', header)[0]
            if not 1 <= size <= 1024:
                continue
            pcm = stream.read(size * 2)
            if len(pcm) != size * 2:
                continue
            take = min(size, target - count)
            wav.writeframesraw(pcm[:take * 2])
            count += take
            deadline = time.monotonic() + timeout
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--port', required=True)
    parser.add_argument('-b', '--baud', type=int, default=921600)
    parser.add_argument('-d', '--duration', type=float, default=20)
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.duration <= 3600:
        parser.error('Duração deve estar em (0, 3600] segundos')
    if args.output.exists():
        parser.error('O arquivo de saída já existe; escolha outro nome de sessão')
    import serial
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.wav')
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as stream:
            stream.reset_input_buffer()
            n = record(stream, partial, args.duration)
        partial.replace(args.output)
        print(f'{args.output}: {n} amostras, {n/16000:.2f} s')
    except (OSError, TimeoutError) as exc:
        parser.exit(1, f'{exc}; captura parcial em {partial}\n')


if __name__ == '__main__':
    main()
