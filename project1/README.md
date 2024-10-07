
# FYS-STK3155: Regression Analysis and Resampling Techniques in Applied Machine Learning

## Abstract

In this paper, we analyze three different methods for polynomial regression in machine learning: Ordinary Least Squares, Ridge, and Lasso Regression. We evaluate their performance with mean squared error and $R^2$-score, and we explore data resampling techniques such as bootstrapping and $\textit{k}$-fold cross validation. First we fit the models to a two dimensional test function, and then to real mountain data from Norway. We find the Ordinary Least Squares to outperform both Ridge and Lasso in approximating real landscapes.

## Installation
> **Note:** Running `main.py` is computationally heavy. 

Clone the repository and install dependencies:

```bash
git clone https://github.com/EOH-ML/FYS-STK3155-Projects.git
cd FYS-STK3155-Projects/project1/src
pip install -r requirements.txt
```

## Usage

To generate all the figures run: 

```bash
python3 main.py
```

Results will be saved in the `figures/` folder.

## Contributors

- **Oscar Brovold** [GitHub](https://github.com/oscarbrovold)
- **Eskil Grinaker Hansen** [GitHub](https://github.com/eskilgrin)
- **Håkon Ganes Kornstad** [GitHub](https://github.com/hakonko)
