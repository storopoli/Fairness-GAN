# Fairness-GAN
Fairness Project using GANs to generate fair data representations applied to the Brazilian context

## Usage

### Python Virtual Environment

1. Install `venv`

   ```bash
   pip install venv
   ```

2. Activate Virtual Environment

   ```bash
   source venv/bin/activate
   ```

3. Instal requirements

   ```bash
   pip install -r requirements.txt
   ```

### Anaconda

1. Create a anaconda environtment from `environment.yml`

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the new environment

   ```bash
   conda activate pytorch
   ```

3. Run the main.py

   ```bash
   chmod +x main.py
   ./main.py --help
   ```

## Fairness

In order to get a 'quantitative' measure of how fair our classifier is, we take inspiration from the U.S. Equal Employment Opportunity Commission (EEOC). They use the so-called [80% rule](https://en.wikipedia.org/wiki/Disparate_impact#The_80%_rule) to quantify the disparate impact on a group of people of a protected characteristic. Zafar et al. show in their paper ["Fairness Constraints: Mechanisms for Fair Classification"](https://arxiv.org/pdf/1507.05259.pdf) how a more generic version of this rule, called the *p%*-rule, can be used to quantify fairness of a classifier. This rule is defined as follows:

> A classifier that makes a binary class prediction <img src="https://render.githubusercontent.com/render/math?math=\hat{y} \in \{0,1\}"> given a binary sensitive attribute <img src="https://render.githubusercontent.com/render/math?math=z \in \{0,1\}"> satisfies the *p\%*-rule if the following inequality holds:

> <img src="https://render.githubusercontent.com/render/math?math=\min \left(\frac{P\left(y^{\wedge}=1 | z=1\right)}{P\left(y^{\prime}=1 | z=0\right)}, \frac{P\left(y^{\wedge}=1 | z=0\right)}{P\left(y^{\prime}=1 | z=1\right)}\right) \geq \frac{p}{100}">

The rule states that the ratio between the probability of a positive outcome given the sensitive attribute being true and the same probability given the sensitive attribute being false is no less than *p*:100. So, when a classifier is completely fair it will satisfy a 100%-rule. In contrast, when it is completely unfair it satisfies a 0%-rule.

In determining the fairness our or classifier we will follow the EEOC and say that a model is fair when it satisfies at least an 80%-rule. So, let's compute the *p%*-rules for the classifier and put a number on its fairness. Note that we will threshold our classifier at 0.5 to make its prediction it binary.

## Neural Networks

The Fairness-GAN is composed of 4 neural networks:

1. **Encoder**: that takes the maping <img src="https://render.githubusercontent.com/render/math?math=X"> to <img src="https://render.githubusercontent.com/render/math?math=\tilde{X}">
2. **Decoder**: that reconstruct <img src="https://render.githubusercontent.com/render/math?math=\tilde{X}"> to <img src="https://render.githubusercontent.com/render/math?math=X">
3. **Classifier**: that tries to predict <img src="https://render.githubusercontent.com/render/math?math=Y"> from <img src="https://render.githubusercontent.com/render/math?math=\tilde{X}">, by mapping <img src="https://render.githubusercontent.com/render/math?math=\tilde{X}"> to <img src="https://render.githubusercontent.com/render/math?math=\hat{Y}">
4. **Discriminator**: that tries to predict if <img src="https://render.githubusercontent.com/render/math?math=\tilde{X}"> has <img src="https://render.githubusercontent.com/render/math?math=Z">

### Encoder

Bla Bla Bla

### Decoder

Bla Bla Bla

### Classifier

Bla Bla Bla

### Discriminator

* Loss function: <img src="https://render.githubusercontent.com/render/math?math=min_{\theta_{clf}} \left( Loss_{y} \left( \theta_{clf} \right) - \lambda Loss_{Z} \left( \theta_{clf}, \theta_{adv} \right) \right)">
* <img src="https://render.githubusercontent.com/render/math?math=\lambda">: regularizer that forces the classifier towards fairer predictions while sacrificing prediction accuracy

## Results

Decoder Error

ROC Curve Classifier

ROC Curve Discriminator
