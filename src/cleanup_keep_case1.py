import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEG = os.path.join(ROOT, 'runs', 'segment')

# Files we want to keep (relative to workspace root)
keep = set([
    os.path.normpath(os.path.join('src', 'runs', 'segment', 'predict', 'WIN_20260610_09_33_53_Pro_watchcase_predict.png')),
    os.path.normpath(os.path.join('src', 'runs', 'segment', 'predict', 'pixels', 'WIN_20260610_09_33_53_Pro_watchcase_pixels.csv')),
])

# Also keep this loader and this cleanup script
keep_scripts = set([
    os.path.normpath(os.path.join('src', 'load_case1_output.py')),
    os.path.normpath(os.path.join('src', 'cleanup_keep_case1.py')),
])

def remove_unless_keep(path):
    rp = os.path.normpath(path)
    # workspace-relative
    rel = os.path.relpath(rp, ROOT)
    rel = os.path.normpath(os.path.join('src', rel))
    if rel in keep or rel in keep_scripts:
        return
    try:
        if os.path.isfile(rp) or os.path.islink(rp):
            os.remove(rp)
            print('Removed file', rp)
        elif os.path.isdir(rp):
            shutil.rmtree(rp)
            print('Removed dir', rp)
    except Exception as e:
        print('Error removing', rp, e)

# Walk runs/segment and remove everything except the two files
for root, dirs, files in os.walk(SEG):
    for f in files:
        p = os.path.join(root, f)
        remove_unless_keep(p)
    for d in dirs:
        p = os.path.join(root, d)
        # if dir becomes empty, it will be removed above when iterating

# Remove other scripts in src/ except loader and cleanup
SRC = os.path.join(ROOT, 'src')
for f in os.listdir(SRC):
    p = os.path.join(SRC, f)
    rel = os.path.normpath(os.path.join('src', f))
    if rel in keep_scripts:
        continue
    if os.path.isfile(p) and f.endswith('.py'):
        try:
            os.remove(p)
            print('Removed script', p)
        except Exception as e:
            print('Error removing script', p, e)

print('\nCleanup complete. Minimal structure kept:')
print(' -', os.path.join('src', 'runs', 'segment', 'predict', 'WIN_20260610_09_33_53_Pro_watchcase_predict.png'))
print(' -', os.path.join('src', 'runs', 'segment', 'predict', 'pixels', 'WIN_20260610_09_33_53_Pro_watchcase_pixels.csv'))
print(' -', os.path.join('src', 'load_case1_output.py'))
