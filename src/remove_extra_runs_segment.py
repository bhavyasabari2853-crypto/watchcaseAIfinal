import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEG = os.path.join(ROOT, 'src', 'runs', 'segment')

KEEP = {'predict'}

if not os.path.exists(SEG):
    print('No runs/segment folder found')
    raise SystemExit(0)

for name in os.listdir(SEG):
    p = os.path.join(SEG, name)
    if name in KEEP:
        print('Keeping', p)
        continue
    try:
        if os.path.isdir(p):
            shutil.rmtree(p)
            print('Removed', p)
        else:
            os.remove(p)
            print('Removed file', p)
    except Exception as e:
        print('Error removing', p, e)

print('Done cleaning runs/segment; kept:', ','.join(KEEP))
