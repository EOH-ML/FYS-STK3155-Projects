# Less is More: Comparing Neural Networks and Regression for Classification and Approximation

## Abstract
Artificial neural networks offer flexibility for solving a variety of prediction and decision problems. However, they can also be computationally costly and energy-intensive. While advances in computational power have brought Machine Learning forward, challenges
remain on the energy consumption side. In this paper, we aim to investigate how a
feed-forward neural network (FFNN) compares to traditional regression methods, which
are more lightweight in terms of computation and easier to implement. We start by
coding a FFNN from scratch and train it using custom optimizers, including Stochastic
Gradient Descent (SGD), RMSProp, and other variants. We approximate synthetic
data generated from Franke’s Function, and then apply the model to classify real health
conditions using the well-documented Breast Cancer Wisconsin (Diagnostic) dataset.
We then implement a logistic regression model for comparison. Our findings show that
a well-tuned FFNN can effectively model both problems. For Franke’s Function, our
optimized Neural Network is able to achieve an MSE of approximately 0.01195. This is
comparable to results from previous testing with an analytic linear regression model.
For classification with the breast cancer dataset, we achieve an accuracy of around
0.982. The logistic regression model provides slightly better accuracy in comparison. We
conclude that traditional methods are preferable for some classification or approximation
tasks. They can produce comparable results to neural networks, while possibly offering
lower computational costs, reduced energy consumption, and simpler implementation.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/EOH-ML/FYS-STK3155-Projects.git
cd FYS-STK3155-Projects/project2/src
pip install -r requirements.txt
```
> **Note:** If the command doesn't run, try replacing `pip` with `pip3`.

## Usage

To generate all the figures and run tests: 

```bash
python3 run_all.py
```

Results will be saved in a `project2/figures/` folder.

There is also an alternative run that keeps track of the power usage of all the test runs combined, using `CodeCarbon` to track your CO2. Note: `CodeCarbon` might require your system password to keep track of your CPU, GPU and RAM's power usage. Can only be run with `python3`:
```bash
python3 run_all_emissions.py
```

## Contributors

- **Oscar Brovold** [GitHub](https://github.com/oscarbrovold)
- **Eskil Grinaker Hansen** [GitHub](https://github.com/eskilgrin)
- **Håkon Ganes Kornstad** [GitHub](https://github.com/hakonko)

