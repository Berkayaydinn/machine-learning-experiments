# Experiment Methods, Results, and Interpretations

This document summarizes information present in the **original uploaded notebooks**. Figures are saved run outputs, **not new training runs**.

## Gaussian Naive Bayes on CIFAR-10

- **Question:** Can class-conditional per-pixel Gaussian statistics distinguish visually similar animals?
- **Classes (original CIFAR-10 IDs):** Horse (7), Deer (4), Dog (5).
- **Samples:** First 5,000 training and 100 test images selected for each class from CIFAR-10's existing train/test datasets. No additional random subsampling is performed.
- **Features:** Normalized flattened RGB images: 3,072 values per sample.
- **Model:** For each class, feature-wise mean and variance (`variance + 1e-6`) and equal observed priors (1/3); manual Gaussian log-likelihood summed across pixels, followed by argmax.
- **Evaluation:** 146/300 correct = 48.67%. Horse 50/100; Deer 56/100; Dog 40/100. The notebook plots a confusion matrix.
- **Limitations:** Conditional feature independence is restrictive for natural images. No alternate feature extraction, calibration, additional random seeds or stronger model baselines are evaluated here.

## Logistic Regression from Scratch on Iris

- **Question:** How do sigmoid, binary cross-entropy and analytical gradients work together in gradient descent?
- **Dataset:** UCI Iris dataset loaded over HTTPS within the notebook; two classes, Iris-versicolor = 0, Iris-virginica = 1.
- **Inputs:** Petal length and petal width only; 100 total samples (50 per class).
- **Split:** Per-class shuffle with NumPy generator seed 42, 40/class training and 10/class testing; 80/20 total.
- **Training:** Custom NumPy implementation with zero-initialized weights and bias, learning rate 0.1, 1,000 epochs, sigmoid, binary cross-entropy and batch gradient descent. Classification threshold 0.5.
- **Saved outputs:** Training loss declines from 0.693147 to 0.3582; final training accuracy 91.25%. **Test accuracy 95.00% = 19/20 correct.**
- **Limitations:** Only 20 held-out examples, one random split and two features; no cross-validation or multiple seeds. The result should not be presented as a broad comparison with trained library implementations.

## LeNet-style CNN vs Mini-VGG on CIFAR-10

- **Question:** Does using a deeper stack of small convolution filters outperform a simpler CNN under matched training conditions?
- **Inputs:** CIFAR-10 32×32 RGB, 10 classes. Training 50,000, held-out test 10,000.
- **Preprocessing:** `ToTensor()` and channel normalization, no data augmentation.
- **Setup:** PyTorch, seed 42, 3-fold stratified cross-validation, 8 epochs per fold, batch size 128, SGD with momentum, cross-entropy loss, learning rate 0.02 for SGD, dropout 0.2, `num_workers=2`. Saved Colab metadata indicates a GPU/T4 setup; notebook output prints `cuda`.
- **Networks:** LeNet-style (two convolution layers, 47,530 parameters); Mini-VGG (four convolution layers in two blocks, 50,138 parameters).
- **Cross-validation:** Mean validation 68.07% (LeNet) vs 67.78% (Mini-VGG). Best single folds: 68.82% and 68.87%, respectively.
- **Test:** Best-validation-fold checkpoint for each network evaluated on the 10,000-image held-out test set: 68.65% (LeNet), 69.24% (Mini-VGG). Difference: 0.59 percentage points. Most frequent error of the selected Mini-VGG: dog predicted as cat (238 test images).
- **Follow-up experiments:** On the Mini-VGG selected validation fold, width 0.5× / 1× / 2× yielded 64.36% / 68.87% / 72.07% validation accuracy. Dropout 0.0 / 0.2 / 0.4 yielded 68.48% / 68.87% / 67.81%. SGD 68.87% versus Adam 65.92%; these use different learning rates (0.02 vs 0.001).
- **Evaluation caution:** The notebook selects the architecture for follow-up experiments using higher test accuracy. This means the test data influenced experimental decision-making. Interpret post-selection experiments as exploratory, not as an unbiased final untouched-test evaluation. A better future protocol would select architecture/hyperparameters exclusively on training/validation, then evaluate once on the test set.
- **Further limitations:** Eight training epochs, one seed, no augmentation, only one validation fold for follow-ups, and different optimizer learning rates. Saved wall-clock durations are environment-dependent.

## Ford Ranger MSRP custom-loss regression

- **Question:** How do raw calendar-year features and standardization with learning-rate decay affect regression under a quartic-augmented loss?
- **Provenance:** Source notebook says 19 annual minimum/maximum Ford Ranger MSRP observations from Cars.com for 2000–2011 and 2019–2025; no independent source/price verification performed in this review.
- **Model:** Separate linear `min` and `max` regressors, each predicting `theta0 + theta1 * year`. Cost `mean(e² + 0.1e⁴)` and analytically derived gradient `(2e + 0.4e³)` multiplied by each parameter's input derivative.
- **Four training configurations:** Unscaled minimum, unscaled maximum, standardized-year minimum with decaying learning rate, standardized-year maximum with decaying learning rate. Each configuration uses 100 iterations. Training losses are not directly comparable across **different target series** (min vs max).
- **Reported final training costs:** Unscaled MIN ≈ 3.9233e14; scaled MIN ≈ 4.1115e13; unscaled MAX ≈ 2.6212e15; scaled MAX ≈ 8.5100e13.
- **Predictions:** Hypothetical missing-year estimates for 2012–2018 and an illustrative **2026** min/max range of **$27,646.67–$49,982.29**. These are not measured prices.
- **Limitations:** No train/test split, small series, only a single feature (year), no price-level adjustment for inflation or specification changes, and quartic loss makes absolute values large. The claimed improvement is in training optimization, not demonstrated out-of-sample prediction quality.
- **Attribution:** Jointly credited to Berkay Aydin and Elif Kanik. The code does not provide a reliable per-contributor breakdown.
