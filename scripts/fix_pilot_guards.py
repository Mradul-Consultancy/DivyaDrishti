"""
Fix cell 22 (IndentationError on with open) and guard cells 30, 32 from auto-cascading.
"""
import json

nb_path = 'notebooks/kaggle_pilot.ipynb'

with open(nb_path) as f:
    nb = json.load(f)

cells = nb['cells']

# --- FIX CELL 22: IndentationError ---
c22_src = ''.join(cells[22]['source'])
target_broken = """with open(summary_path) as f:
# ---- Quality gate: check VALIDATION AUC (configurable via env-var) ----"""

target_fixed = """with open(summary_path) as f:
    summary = json.load(f)

# ---- Quality gate: check VALIDATION AUC (configurable via env-var) ----"""

if target_broken in c22_src:
    c22_src = c22_src.replace(target_broken, target_fixed)
    cells[22]['source'] = [c22_src]
    print("Fixed cell 22: Added 'summary = json.load(f)' inside with open block")
else:
    # Try finding with open(summary_path) as f: without json.load(f)
    print("Warning: target_broken pattern not matched exactly, checking lines...")
    lines = cells[22]['source']
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if 'with open(summary_path) as f:' in line:
            new_lines.append('    summary = json.load(f)\n\n')
    cells[22]['source'] = new_lines
    print("Added summary = json.load(f) after with open")

# --- GUARD CELL 30: Don't cascade if pilot=True ---
c30_src = ''.join(cells[30]['source'])
if 'if not pilot:' not in c30_src and 'if pilot:' not in c30_src:
    guarded_c30 = """import types

if pilot:
    print('🛑 PILOT MODE: Skipping full training grid as per protocol.')
    print('   Inspect pilot_outputs.zip, training curves, and gate output before running full grid.')
else:
    from scripts.train_all_datasets import run_all

    full_args = types.SimpleNamespace(
        datasets   = None,            # None = all available datasets (celebdf, dfdc)
        epochs     = FULL_EPOCHS,
        batch_size = FULL_BATCH,
        patience   = FULL_PATIENCE,
        pilot      = False,
        skip_yolo  = SKIP_YOLO,
        force      = FORCE_RETRAIN,
    )

    print(f'Starting full training grid: {FULL_EPOCHS} epochs, patience {FULL_PATIENCE}...')
    run_all(full_args)
    print('✅ Training grid complete.')
"""
    cells[30]['source'] = [guarded_c30]
    print("Guarded cell 30: Only runs full training grid when pilot=False")

# --- GUARD CELL 32: Don't run cross-table if pilot=True ---
c32_src = ''.join(cells[32]['source'])
if 'if not pilot:' not in c32_src and 'if pilot:' not in c32_src:
    guarded_c32 = """import types

if pilot:
    print('🛑 PILOT MODE: Skipping cross-dataset evaluation until full grid checkpoints exist.')
else:
    from scripts.build_cross_table import run_cross_table

    ct_args = types.SimpleNamespace(
        batch_size = 32,
        device     = str(config.DEVICE),
    )

    results = run_cross_table(ct_args)
    print('✅ Cross-dataset evaluation table complete.')
"""
    cells[32]['source'] = [guarded_c32]
    print("Guarded cell 32: Only runs cross-table when pilot=False")

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Saved updated notebook.")
