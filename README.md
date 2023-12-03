CUDA 11.8:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium
pip install highway-env
pip install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
pip install stable_baselines3[extra]

change import in stable_baselines3 gym to gymnasium