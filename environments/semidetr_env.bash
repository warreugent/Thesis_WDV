# install the .tar file via github: https://github.com/JCZ404/Semi-DETR/blob/main/README.md
# put it into the envs directory of your anaconda/miniconda, where anaconda/miniconda manage their virtual envs. Then unzip this file, and execute conda init to make the env prepared.

# once the env is prepared, clone the github repo 
cd ~/Thesis_WDV/External
git clone https://github.com/JCZ404/Semi-DETR.git

# Activate the env
conda activate semidetr

# Add NVIDIA channel and install cudart 12.1 + add it to the path
conda install -c nvidia cuda-cudart=12.1.105
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

# 1) mmdetection (editable install)
cd ~/Thesis_WDV/External/Semi-DETR/thirdparty/mmdetection
python -m pip install -e .

# 2) Semi-DETR packages (detr_od, detr_ssod) – editable
cd ~/Thesis_WDV/External/Semi-DETR
python -m pip install -e .

# 3) Build deformable attention CUDA ops (once)
cd detr_od/models/utils/ops
python setup.py build install
# optional sanity check (out of memory on the A10 GPU)
python test.py
cd ../../../..