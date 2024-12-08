# Sleep Scoring in Alzheimer Mouse Models: Comparing Traditional and Spatial Approaches using Feed Forward and Convolusional Neural Networks

## Abstract
It is established that sleep plays a role in Alzheimer’s Disease, influencing processes
that may impact disease progression. Sleep scoring, vital for understanding these
mechanisms, is typically labor-intensive and performed manually. Automating sleep
scoring with machine learning is therefore potentially a great tool for improving the
efficiency in Alzheimer research. We build two machine learning models for sleep
scoring in mice based on the same dataset of 14 different test mouse models from trials
measuring brainwaves with electrocorticography (ECoG) and muscle activities with
Electromyography (EMG). We train our Feed Forward Neural Network (FFNN) in
tandem with an extensive feature engineering, yielding an overall accuracy of 87%,
as well as 67% and 48% on IS and REM, respectively. Then we compare it to a
CNN-model with two convolution layers, kernel size (2, 2) and MaxPooling, using
ReLU as activation function in the dense layers. It is trained on plots of ECoG and
EMG in the frequency domain, and yields an overall accuracy of 92%, and sleep stage
individual F1-scores of 74% and 69% on Intermediate Sleep (IS) and REM, respectively.
We conclude that while FFNNs can effectively learn diverse tasks, the CNN outperforms
it in our testing. We propose this is due to the CNN’s ability to extract spatial patterns
from frequency-domain data, making it better suited for sleep scoring.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/EOH-ML/FYS-STK3155-Projects.git
cd FYS-STK3155-Projects/project3/src
pip install -r requirements.txt
```
> **Note:** If the command doesn't run, try replacing `pip` with `pip3`.

## Usage

To generate all the figures and run tests: 

```bash
python3 run_all.py
```

Results will be saved in a `project3/figures/` folder.

## Contributors

- **Oscar Brovold** [GitHub](https://github.com/oscarbrovold)
- **Eskil Grinaker Hansen** [GitHub](https://github.com/eskilgrin)
- **Håkon Ganes Kornstad** [GitHub](https://github.com/hakonko)
