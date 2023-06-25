
Run ``pip install -r requirements.txt``  or build manually: 
<!-- cat requirements.txt | xargs -n 1 pip install -->

```
conda create --name disclip python=3.8
conda activate disclip
conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge transformers==4.7.0
conda install -c conda-forge ftfy
conda install -c anaconda pandas 
conda install -c anaconda chainer
conda install -c anaconda scikit-image 
conda install -c conda-forge matplotlib
conda install -c conda-forge sacrebleu==1.4.10
conda install -c conda-forge spacy  # reclip 
conda install -c anaconda sqlalchemy  # reclip
conda install -c conda-forge ruamel.yaml  # reclip
conda install -c conda-forge timm  # reclip
conda install -c conda-forge overrides # reclip
pip install cupy-cuda11x
python -m cupyx.tools.install_library --library nccl --cuda 11.x

```