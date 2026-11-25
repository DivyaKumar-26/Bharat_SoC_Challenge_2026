@echo off
REM Windows: full Phase 1 pipeline. Record the screen (Win+Alt+R) during the last step.
pip install -r requirements.txt || exit /b 1
python download_data.py || exit /b 1
python qc_assistant.py fit  --normal mvtec/bottle/train/good --out bank.pt > fit.txt
type fit.txt
python qc_assistant.py eval --bank bank.pt --test mvtec/bottle/test > results.txt
type results.txt
echo Starting live demo - press q to stop
python qc_assistant.py run --bank bank.pt --source mvtec/bottle/test
