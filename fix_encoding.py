# fix_encoding.py — repairs non-ASCII chars garbled during PowerShell concat
import re

# Read raw bytes, detect actual encoding
with open('app.py', 'rb') as f:
    raw = f.read()

# Try to decode; if UTF-8 fails, it was re-encoded as latin-1 by PowerShell
try:
    content = raw.decode('utf-8')
    # Check if the UTF-8 was double-encoded (latin-1 of UTF-8 bytes)
    if 'â€' in content or 'ðY' in content or 'Â·' in content:
        # It was read as latin-1 then written as UTF-8 — double decode to fix
        content = raw.decode('utf-8').encode('latin-1').decode('utf-8')
        print("Fixed double-encoding issue")
    else:
        print("UTF-8 looks clean")
except (UnicodeDecodeError, UnicodeEncodeError):
    # Raw bytes are latin-1 / Windows-1252 encoded UTF-8
    content = raw.decode('latin-1').encode('latin-1').decode('utf-8', errors='replace')
    print("Fixed latin-1 encoding")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Saved. Verifying...")
import ast
ast.parse(content)
print("Syntax OK. Non-ASCII count:", sum(1 for c in content if ord(c) > 127))
