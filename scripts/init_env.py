"""Create local credentials once; never print or replace existing secrets."""
import os
import secrets
from pathlib import Path

target = Path('.env')
if target.exists():
    print('.env already exists; preserved')
else:
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as file:
        file.write(f'ADMIN_API_KEY={secrets.token_urlsafe(32)}\n')
        file.write(f'GRAFANA_ADMIN_PASSWORD={secrets.token_urlsafe(24)}\n')
    print('Created .env with local credentials (mode 0600)')
