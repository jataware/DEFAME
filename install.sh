

conda create -n infact_env python=3.10
conda activate infact_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install -c conda-forge numpy pandas scikit-learn scipy matplotlib nltk tqdm setuptools jinja2 requests pyparsing
conda install -c pytorch -c nvidia faiss-gpu=1.9.0

pip install bitsandbytes==0.41.3
pip install duckduckgo_search==6.1.0
pip install huggingface-hub>=0.23.2
pip install langdetect>=1.0.9
pip install openai>=1.37.0
pip install orjsonl==1.0.0
pip install sentence-transformers
pip install sty>=1.0.6
pip install tiktoken~=0.7.0
pip install transformers>=4.44.2
pip install rich
pip install kagglehub