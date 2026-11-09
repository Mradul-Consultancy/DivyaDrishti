# run.ps1 — PowerShell make-replacement for Deepfake Detection pipeline
# Usage: .\run.ps1 -target pilot|full|reports|real|deliver|clean
# No GNU Make required.

param(
    [ValidateSet('pilot','full','reports','real','deliver','clean','test','help')]
    [string]$target = 'help',
    [int]$MaxVideos   = 200,
    [int]$FullEpochs  = 25,
    [string]$Env      = 'LOCAL',
    [float]$QualityGate = 0.70
)

$Env:DEEPFAKE_ENV = $Env
$Env:QUALITY_GATE = "$QualityGate"

function Show-Help {
    Write-Host @"
Deepfake Detection Pipeline (PowerShell Runner)

Usage:
  .\run.ps1 -target <TARGET> [options]

Available Targets:
  test     - Run the automated test suite (pytest tests/)
  pilot    - Run pilot notebook (5 epochs, MAX_VIDEOS=200, checks val_roc_auc >= QUALITY_GATE)
  full     - Run full training grid (25 epochs, all models/datasets; manual step)
  reports  - Evaluate ensemble and generate evaluation reports
  real     - Build real-life image verification set (data/real_life/)
  deliver  - Regenerate data-only FINAL_CLIENT_DELIVERABLE.md (manual step)
  clean    - Remove transient notebooks and temporary zip files
  help     - Show this help message

Options:
  -MaxVideos <int>    Max videos to process for pilot (default: 200)
  -FullEpochs <int>   Epochs for full training grid (default: 25)
  -QualityGate <float> Validation AUC threshold for quality gate (default: 0.70)
  -Env <string>       Execution environment: LOCAL or KAGGLE (default: LOCAL)

Note: As per delivery constraints, 'full' and 'deliver' are deliberate manual steps.
"@ -ForegroundColor Cyan
}

function Run-Test {
    Write-Host "`n=== Running full test suite ===" -ForegroundColor Cyan
    python -m pytest tests/ -q
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Test suite failed."
        exit 1
    }
    Write-Host "✅ All tests passed." -ForegroundColor Green
}

function Run-Pilot {
    Write-Host "`n=== Pilot: face extraction & split ===" -ForegroundColor Cyan
    papermill notebooks/kaggle_pilot.ipynb pilot_run.ipynb `
        -p MAX_VIDEOS $MaxVideos `
        -p epochs 5 `
        -p pilot 1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Pilot notebook failed — check pilot_run.ipynb for details."
        exit 1
    }
    Write-Host "✅ Pilot complete." -ForegroundColor Green
}

function Run-Full {
    Write-Host "`n=== Full run: training all models (Manual step, no auto-cascade) ===" -ForegroundColor Cyan
    papermill notebooks/kaggle_pilot.ipynb full_run.ipynb `
        -p MAX_VIDEOS "None" `
        -p epochs $FullEpochs `
        -p pilot 0
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Full-run notebook failed — check full_run.ipynb."
        exit 1
    }
    Write-Host "✅ Full run complete." -ForegroundColor Green
}

function Run-Reports {
    Write-Host "`n=== Evaluating ensemble and generating reports ===" -ForegroundColor Cyan
    python main.py evaluate-ensemble
    python main.py generate-reports
    if ($LASTEXITCODE -ne 0) { Write-Error "Report generation failed."; exit 1 }
    Write-Host "✅ Reports written to reports/" -ForegroundColor Green
}

function Run-Real {
    Write-Host "`n=== Building real-life image set ===" -ForegroundColor Cyan
    python scripts/build_real_life_dataset.py --output-dir data/real_life --num-images 2000
    if ($LASTEXITCODE -ne 0) { Write-Error "Real-life dataset build failed."; exit 1 }
    Write-Host "✅ Real-life image set ready." -ForegroundColor Green
}

function Run-Deliver {
    Write-Host "`n=== Regenerating FINAL_CLIENT_DELIVERABLE.md ===" -ForegroundColor Cyan
    python scripts/regenerate_report.py
    if ($LASTEXITCODE -ne 0) { Write-Error "Report regeneration failed."; exit 1 }
    Write-Host "✅ Client deliverable updated." -ForegroundColor Green
}

function Run-Clean {
    Write-Host "`n=== Cleaning intermediate artefacts ===" -ForegroundColor Cyan
    Remove-Item -ErrorAction SilentlyContinue pilot_run.ipynb, full_run.ipynb, *.zip
    Write-Host "✅ Clean complete." -ForegroundColor Green
}

switch ($target) {
    'test'    { Run-Test    }
    'pilot'   { Run-Pilot   }
    'full'    { Run-Full    }
    'reports' { Run-Reports }
    'real'    { Run-Real    }
    'deliver' { Run-Deliver }
    'clean'   { Run-Clean   }
    'help'    { Show-Help   }
}

