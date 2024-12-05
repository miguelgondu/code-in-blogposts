# Introduction

# Setting up the env

```bash 
conda create -n additive-rasp-blogpost python=3.11
# pystan is picky, and it needs numpy and
# Cython to already be there.
pip install numpy Cython
pip install -r requirements.txt
```

# Downloading the necessary files.

- [RFP pool data from LaMBO](https://github.com/samuelstanton/lambo/blob/main/lambo/assets/fpbase/proxy_rfp_seed_data.csv): `curl https://raw.githubusercontent.com/samuelstanton/lambo/main/lambo/assets/fpbase/proxy_rfp_seed_data.csv -O`


# Time taken for RaSP to run on ~2000k mutations

- `Time taken: 770.57s`
- Computing RaSP twice renders different values. This might be because we're using single precision. This is something that pops up in our `poli` tests too.