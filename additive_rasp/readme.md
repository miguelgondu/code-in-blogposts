```bash 
conda create -n additive-rasp-blogpost python=3.11
# pystan is picky, and it needs numpy and
# Cython to already be there.
pip install numpy Cython
pip install -r requirements.txt
```