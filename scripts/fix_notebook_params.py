"""
Fix notebook parameter bugs:
1. Add None-casting cell after params cell
2. Remove hardcoded MAX_VIDEOS=200 from download cell (use param)
3. Replace hardcoded epochs=5 in training cell with param
"""
import json
import sys

nb_path = 'notebooks/kaggle_pilot.ipynb'

with open(nb_path) as f:
    nb = json.load(f)

cells = nb['cells']
changes = []

# --- FIX 1: Check if None-casting cell already inserted ---
cast_marker = 'MAX_VIDEOS = None if'
already_has_cast = any(cast_marker in ''.join(c['source']) for c in cells)

if not already_has_cast:
    cast_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["injected-parameters"]},
        "outputs": [],
        "source": [
            "# \u2500\u2500 Parameter type-casting (papermill passes everything as strings) \u2500\u2500\n",
            "# When papermill injects MAX_VIDEOS=None it arrives as the string 'None'\n",
            "MAX_VIDEOS = None if str(MAX_VIDEOS).strip() == 'None' else int(MAX_VIDEOS)\n",
            "epochs = int(epochs)\n",
            "pilot  = bool(int(pilot))\n",
            "print(f'Parameters: MAX_VIDEOS={MAX_VIDEOS!r}  epochs={epochs}  pilot={pilot}')\n"
        ]
    }
    # Insert right after cell 0 (params cell)
    cells.insert(1, cast_cell)
    changes.append('Inserted None-casting cell after params cell (now cell 1)')
else:
    changes.append('None-casting cell already present - skipped')

# After insertion, re-index: params=0, cast=1, rest shift by 1
# Cell 7 was download, now it's cell 8 if we inserted one
# Let's find cells by content instead

# --- FIX 2: Find download cell and remove hardcoded MAX_VIDEOS ---
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    # The hardcoded line in the download cell
    if 'MAX_VIDEOS = 200' in src and 'download_dfdc_from_dataset' in src:
        old_line = 'MAX_VIDEOS = 200     # \u2190 tweak: 50 for a 10-min smoke-test, 200 for a real pilot\n'
        new_line = '# MAX_VIDEOS comes from the parameters cell (cast None-safe by cell 1)\n'
        new_src = src.replace(old_line, new_line)
        if new_src != src:
            cells[i]['source'] = [new_src]
            changes.append(f'Cell {i}: removed hardcoded MAX_VIDEOS=200 from download cell')
        else:
            # Try without trailing newline
            old_line2 = 'MAX_VIDEOS = 200     # \u2190 tweak: 50 for a 10-min smoke-test, 200 for a real pilot'
            new_src2 = src.replace(old_line2, '# MAX_VIDEOS comes from the parameters cell (cast None-safe by cell 1)')
            if new_src2 != src:
                cells[i]['source'] = [new_src2]
                changes.append(f'Cell {i}: removed hardcoded MAX_VIDEOS=200 from download cell (no trailing newline)')
            else:
                changes.append(f'Cell {i}: WARNING - could not find MAX_VIDEOS=200 to replace')
        break

# --- FIX 3: Find training cell and replace hardcoded epochs=5 ---
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    if 'run_timm_training' in src and ('epochs          = 5,' in src or "epochs          = 5," in src):
        old_line = "    epochs          = 5,            # short pass \u2014 bump to 25 for real run"
        new_line = "    epochs          = int(epochs),  # from parameters cell (pilot=5, full=25)"
        new_src = src.replace(old_line, new_line)
        if new_src != src:
            cells[i]['source'] = [new_src]
            changes.append(f'Cell {i}: replaced hardcoded epochs=5 with int(epochs) param')
        else:
            changes.append(f'Cell {i}: WARNING - could not replace epochs line (maybe already fixed or different whitespace)')
        break

nb['cells'] = cells

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print('Changes applied:')
for ch in changes:
    print(f'  - {ch}')
print('Done.')

