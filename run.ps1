# run.ps1 — PowerShell make-replacement for Deepfake Detection pipeline
# Usage: .\run.ps1 -target pilot|full|reports|real|deliver|clean
# No GNU Make required.

param(
    [ValidateSet('pilot','full','reports','real','deliver','clean')]
    [string]$target = 'pilot',
    [int]$MaxVideos   = 200,
    [int]$FullEpochs  = 25,
    [string]$Env      = 'LOCAL',
    [float]$QualityGate = 0.70
)

$Env:DEEPFAKE_ENV = $Env
$Env:QUALITY_GATE = "$QualityGate"

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
    Run-Pilot
    Write-Host "`n=== Full run: training all models ===" -ForegroundColor Cyan
    papermill pilot_run.ipynb full_run.ipynb `
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
    'pilot'   { Run-Pilot   }
    'full'    { Run-Full    }
    'reports' { Run-Reports }
    'real'    { Run-Real    }
    'deliver' { Run-Deliver }
    'clean'   { Run-Clean   }
}

