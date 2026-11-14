import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'ckpt_index.py'
spec = importlib.util.spec_from_file_location('index', SCRIPT)
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)

class IndexTests(unittest.TestCase):
    def test_cli_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Byte-identical CLI copy keeps DEFAULT_INDEX away from real user data.
            script = root/'ckpt_index.py'
            script.write_bytes(SCRIPT.read_bytes())
            (root/'logdir').mkdir()
            current = {'path': str(root), 'config': {'seed': 1}}
            for entries in [{'old': str(root)}, {'new': current},
                            {'old': str(root), 'new': current, 'missing': str(root/'absent')}]:
                with self.subTest(entries=entries):
                    (root/'logdir/ckpt_index.json').write_text(json.dumps(entries))
                    p = subprocess.run([sys.executable, str(script), 'list'], capture_output=True, text=True)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    for name in entries: self.assertIn(name, p.stdout)
                    self.assertNotIn("'config'", p.stdout)
            for entry in [None, 12, {}, {'path': None}, {'path': []}]:
                with self.subTest(malformed=entry):
                    (root/'logdir/ckpt_index.json').write_text(json.dumps({'bad': entry, 'good': str(root)}))
                    p = subprocess.run([sys.executable, str(script), 'list'], capture_output=True, text=True)
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn('Invalid checkpoint entry', p.stderr)
                    self.assertIn('good', p.stdout)

    def test_lookup_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'index.json'
            path.write_text(json.dumps({'old': tmp, 'new': {'path': tmp, 'config': {'seed': 1}}}))
            self.assertEqual(index.resolve_name('old', path), (tmp, {}))
            self.assertEqual(index.resolve_name('new', path), (tmp, {'seed': 1}))
            path.write_text(json.dumps({'gone': {'path': tmp+'/absent'}}))
            with self.assertRaises(FileNotFoundError): index.resolve_name('gone', path)

if __name__ == '__main__': unittest.main()
