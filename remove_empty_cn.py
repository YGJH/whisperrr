from pathlib import Path
import os
current_dir = Path('.')
a = {p.stem for p in current_dir.glob("*.mp4")}
for f in current_dir.glob("*.srt"):
    # print(f.stem.removeprefix('cn_'))
    if f.stem.startswith('cn_') and f.stem.removeprefix('cn_') not in a:
        os.remove(f)
        print(f'delete: {f.name}')
    elif not f.stem.startswith('cn_') and f.stem not in a:
        os.remove(f)
        print(f'delete: {f.name}')
    