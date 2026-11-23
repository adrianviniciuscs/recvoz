import base64
import importlib.util
import io
from pathlib import Path
import struct
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('grab_raw', ROOT / 'tools/grab_raw.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

esp_spec = importlib.util.spec_from_file_location('grab_esp', ROOT / 'tools/grab_esp.py')
esp_module = importlib.util.module_from_spec(esp_spec)
esp_spec.loader.exec_module(esp_module)

col_spec = importlib.util.spec_from_file_location('coletor', ROOT / 'tools/coletor.py')
col_module = importlib.util.module_from_spec(col_spec)
col_spec.loader.exec_module(col_module)


class Stream(io.BytesIO):
    def read_until(self, delimiter):
        result = b''
        while not result.endswith(delimiter):
            byte = self.read(1)
            if not byte:
                break
            result += byte
        return result


class FakeEspSerial:
    def __init__(self, responses):
        self.commands = []
        self._lines = list(responses)

    def write(self, data):
        self.commands.append(data.decode('ascii').strip())

    def flush(self):
        pass

    def readline(self):
        return self._lines.pop(0) if self._lines else b''


class CollectionTests(unittest.TestCase):
    def test_framing_and_exact_duration(self):
        packet = b'\xaa\x55' + struct.pack('<H', 1024) + struct.pack('<h', 1234) * 1024
        stream = Stream(b'boot log\r\n\xaa\x55\x00\x00' + packet * 2)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'audio.wav'
            self.assertEqual(module.record(stream, output, .1), 1600)
            with wave.open(str(output)) as wav:
                self.assertEqual(wav.getnframes(), 1600)
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.readframes(1600), struct.pack('<h', 1234) * 1600)

    def test_grab_esp_plan_and_record_chunks(self):
        self.assertEqual(esp_module.plan_chunks(5), [3, 2])
        self.assertEqual(esp_module.plan_chunks(6), [3, 3])
        pcm1 = struct.pack('<h', 321) * 160
        pcm2 = struct.pack('<h', -123) * 160
        responses = [
            b'log ruido\r\n',
            b'RECVOZ_B64_START 160\r\n',
            base64.b64encode(pcm1) + b'\r\n',
            b'RECVOZ_B64_END\r\n',
            b'RECVOZ_B64_START 160\r\n',
            base64.b64encode(pcm2) + b'\r\n',
            b'RECVOZ_B64_END\r\n',
        ]
        stream = FakeEspSerial(responses)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'esp.wav'
            total = esp_module.record_chunks(stream, output, 5)
            self.assertEqual(total, 320)
            self.assertEqual(stream.commands, ['grab 3', 'grab 2'])
            with wave.open(str(output)) as wav:
                self.assertEqual(wav.getnframes(), 320)
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.readframes(320), pcm1 + pcm2)

    def test_coletor_helpers(self):
        self.assertEqual(col_module.sanitize_name('  João Silva! '), 'joaosilva')
        self.assertEqual(len(col_module.sanitize_name('a' * 30)), 15)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(col_module.next_session_number(folder, 'adrian'), 1)
            (folder / 'locutor_adrian_s1.wav').write_bytes(b'')
            (folder / 'locutor_adrian_s2.wav').write_bytes(b'')
            self.assertEqual(col_module.next_session_number(folder, 'adrian'), 3)
