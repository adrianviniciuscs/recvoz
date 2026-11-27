import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
from recvoz.features import Dataset, load_dataset, voiced_windows, pipeline, noise_profile
from recvoz.evaluation import session_split, closed_set, open_set

ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def test_window_shape_and_silence(self):
        pcm = np.full(32000, 1000, dtype=np.int16)
        windows = list(voiced_windows(pcm))
        self.assertEqual(len(windows), 1)
        self.assertEqual(pipeline(windows[0] / 32768)[0].shape, (51, 126))
        self.assertEqual(list(voiced_windows(np.zeros(32000, dtype=np.int16))), [])
        # Timeout descarta fala insuficiente em uma janela de 10 segundos.
        sparse = np.r_[pcm[:1600], np.zeros(1600 * 99, dtype=np.int16), pcm[:30400]]
        self.assertEqual(list(voiced_windows(sparse)), [])

    def test_no_cross_session_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            for s in [1, 2]:
                sf.write(Path(tmp) / f'locutor_A_s{s}.wav', np.full(16000, .1), 16000)
            with self.assertRaisesRegex(ValueError, 'sem janela'):
                load_dataset(tmp)

    def test_noise_zero_is_passthrough(self):
        y = np.sin(np.arange(32000) * .1) * .2
        a, _ = pipeline(y)
        b, _ = pipeline(y, noise=noise_profile(np.zeros(8192)))
        np.testing.assert_array_equal(a, b)

    def test_session_split_disjoint_and_repeatable(self):
        labels = np.repeat(['A', 'B'], 6)
        groups = np.array([f'{s}/{g}' for s in ['A', 'B'] for g in range(3) for _ in range(2)])
        parts = session_split(labels, groups, calibration=True)
        for i, a in enumerate(parts):
            self.assertEqual(set(labels[a]), {'A', 'B'})
            np.testing.assert_array_equal(a, session_split(labels, groups, calibration=True)[i])
            for b in parts[i+1:]:
                self.assertFalse(set(groups[a]) & set(groups[b]))
        with self.assertRaisesRegex(ValueError, '3 sessões'):
            session_split(labels[:4], groups[:4], calibration=True)

    def test_open_set_protocol_and_metrics(self):
        x = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3, dtype=float)
        data = Dataset(x, np.array(['A']*3+['B']*3), np.array(['A/1','A/2','A/3','B/1','B/2','B/3']), [], [])
        cal = Dataset(np.array([[0, -1, 0.]]), np.array(['C']), np.array(['C/1']), [], [])
        test = Dataset(np.array([[0, 0, -1.]]), np.array(['D']), np.array(['D/1']), [], [])
        result = open_set(data, cal, test)
        self.assertEqual(result['false_accept_rate'], 0)
        self.assertEqual(result['false_reject_rate'], 0)
        with self.assertRaisesRegex(ValueError, 'distintos'):
            open_set(data, cal, cal)
        self.assertEqual(closed_set(data)['accuracy'], 1)
        # Test values never participate in tau selection.
        test.vectors[:] = [0, 1, 0]
        self.assertEqual(open_set(data, cal, test)['tau'], result['tau'])
        self.assertEqual(open_set(data, cal, test)['false_accept_rate'], 1)

    def test_export_dynamic_speakers_and_invalid_input_preserves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); audio = tmp / 'audio'; audio.mkdir()
            y = .2 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000)
            for name in ['A', 'B', 'C']:
                sf.write(audio / f'locutor_{name}_s1.wav', y, 16000, subtype='PCM_16')
            cmd = [sys.executable, str(ROOT / 'tools/export_mfcc.py'), '--audio-dir', str(audio),
                   '--fixture-wav', str(audio / 'locutor_A_s1.wav'), '--output-root', str(tmp / 'out')]
            done = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            header = tmp / 'out/hardware/recvoz-esp/components/speaker_db/enroll_default.h'
            before = header.read_bytes()
            self.assertIn(b'ENROLL_DEFAULT_NSPK 3', before)
            self.assertIn(b'ENROLL_DEFAULT_NVEC 3', before)
            sf.write(audio / 'locutor_D_s1.wav', np.zeros(16000), 16000)
            done = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertEqual(before, header.read_bytes())
            # Paridade C/Python também em sinal sintético, sem áudios privados.
            binary = ROOT / 'tests/host/mfcc/hosttest'
            if binary.exists():
                done = subprocess.run([str(binary), str(tmp / 'out/tests/fixtures') + '/'], capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


if __name__ == '__main__':
    unittest.main()
