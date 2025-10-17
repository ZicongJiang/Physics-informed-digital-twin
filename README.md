<div align="center">
<h1>PIDT: Physics-Informed Digital Twin for Optical Fiber Parameter Estimation</h1>

<a href="" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-blue" alt="Paper PDF">
</a>
<a href=""><img src="https://img.shields.io/badge/arXiv-blue" alt="arXiv"></a>
<a href="/"><img src="https://img.shields.io/badge/project_page-green" alt="Project Page"></a>

**[Department of Electrical Engineering, Chalmers University of Technology](https://www.chalmers.se/en/departments/e2/)**<br>
**[Department of Microtechnology and Nanoscience, Chalmers University of Technology](https://www.chalmers.se/en/departments/mc2/)**

[Zicong Jiang](https://zicongjiang.github.io/zicong-jiang/), [Magnus Karlsson](https://www.chalmers.se/en/persons/magkar/), [Erik Agrell](https://www.chalmers.se/en/persons/agrell/), [Christian Häger](https://chaeger.github.io/)
</div>

````
⚠️ The code will be realsed once the paper is accepted.
````

## Overview

We propose physics-informed digital twin (PIDT): a fiber parameter estimation approach that combines a parameterized split-step method with a physics-informed loss. PIDT improves accuracy and convergence speed with lower complexity compared to previous neural operators.

![framework](assets/Overview.png)

Performance

![results](assets/results.png)


## Install dependencies

```
```

<!-- # step 1: create a new conda environment (tested on Linux)
conda create -n fds python=3.10 pip
# or `conda create --prefix /data/conda/fds python=3.10 pip` if you want to install it in a specific directory

# step 2: activate the environment
conda activate fds
# or `conda activate /data/conda/fds` if you installed it in a specific directory

# step 3: install the dependencies
pip install -r requirements.txt

# step 4: install additional dependencies
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
cd .. -->

### Parameter estimation

####  The PIDT for parameter estimation

```python

```

<!-- python run2d.py # FDS

# which is equivalent to the default setting: python run2d.py --image_path "data/stones.png" --source_prompt "a stack of stone" --target_prompt "a Buddha statue" --dwt_dds --use_dds --J 2 --num_iters 600 --gs 7.5 --seed 24 --keep_low

python run2d.py --use_dds # DDS -->

####  The PINO for parameter estimation

```python

```

---
Thanks for your interest in our work!
If you find our work useful, please consider citing our paper or using this implementation.
```bibtex
@article{
}
```