"""Start script — reads PORT from environment, no shell expansion needed."""
import os
import subprocess
import sys

port = os.environ.get('PORT', '8000')
print(f'[NIDS] Starting gunicorn on port {port}', flush=True)
subprocess.run([
    sys.executable, '-m', 'gunicorn', 'app:app',
    '--bind', f'0.0.0.0:{port}',
    '--workers', '1',
    '--threads', '4',
    '--timeout', '600',
])
