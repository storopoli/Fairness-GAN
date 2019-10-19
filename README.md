# Fairness-GAN
Fairness Project using GANs to generate fair data representations applied to the Brazilian context

## Usage

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

In order to get a 'quantitative' measure of how fair our classifier is, we take inspiration from the U.S. Equal Employment Opportunity Commission (EEOC). They use the so-called [80% rule](https://en.wikipedia.org/wiki/Disparate_impact#The_80%_rule) to quantify the disparate impact on a group of people of a protected characteristic. Zafar et al. show in their paper ["Fairness Constraints: Mechanisms for Fair Classification"](https://arxiv.org/pdf/1507.05259.pdf) how a more generic version of this rule, called the $p\%$-rule, can be used to quantify fairness of a classifier. This rule is defined as follows:

> A classifier that makes a binary class prediction $\hat{y} \in \{0,1\}$ given a binary sensitive attribute $z \in \{0,1\}$ satisfies the $p\%$-rule if the following inequality holds:
> $$\min \left(\frac{P\left(y^{\wedge}=1 | z=1\right)}{P\left(y^{\prime}=1 | z=0\right)}, \frac{P\left(y^{\wedge}=1 | z=0\right)}{P\left(y^{\prime}=1 | z=1\right)}\right) \geq \frac{p}{100}$$

The rule states that the ratio between the probability of a positive outcome given the sensitive attribute being true and the same probability given the sensitive attribute being false is no less than $p$:100. So, when a classifier is completely fair it will satisfy a 100%-rule. In contrast, when it is completely unfair it satisfies a 0%-rule.

In determining the fairness our or classifier we will follow the EEOC and say that a model is fair when it satisfies at least an 80%-rule. So, let's compute the $p\%$-rules for the classifier and put a number on its fairness. Note that we will threshold our classifier at $0.5$ to make its prediction it binary.

## Neural Networks

The Fairness-GAN is composed of 4 neural networks:

1. **Encoder**: that takes the maping $X$ to $\tilde{X}$
2. **Decoder**: that reconstruct $\tilde{X}$ to $X$
3. **Classifier**: that tries to predict $Y$ from $\tilde{X}$, by mapping $\tilde{X}$ to $\hat{Y}$
4. **Discriminator**: that tries to predict if $\tilde{X}$ has $Z$

### Encoder

Bla Bla Bla

### Decoder

Bla Bla Bla

### Classifier

Bla Bla Bla

### Discriminator

* Loss function: $\min _{\theta_{c l f}}\left[\operatorname{Loss}_{y}\left(\theta_{c l f}\right)-\lambda \operatorname{Loss}_{Z}\left(\theta_{c l f}, \theta_{a d v}\right)\right]$

* $\lambda$: regularizer that forces the classifier towards fairer predictions while sacrificing prediction accuracy

## Results

Decoder Error

ROC Curve Classifier

ROC Curve Discriminator