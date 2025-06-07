@echo off
call .venv\Scripts\activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
pause