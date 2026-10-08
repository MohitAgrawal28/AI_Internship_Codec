# AI Internship Project Demos

This repository contains two Streamlit demos built for an AI internship project:

1. **Customer Churn Predictor** - a classical machine learning app for telecom churn risk prediction.
2. **CNN Image Classifier** - a deep learning app that demonstrates image classification with a Keras CNN.

## Projects

### Customer Churn Predictor

The churn predictor uses customer account, service, billing, and demographic fields to estimate whether a customer is likely to leave.

Key ideas:

- Feature engineering for tenure buckets, average spend, and add-on count.
- Column-wise preprocessing with imputation, scaling, and one-hot encoding.
- SMOTE for handling class imbalance.
- Random Forest classifier tuned with cross-validation.
- Recall-oriented decision threshold because missing a churner is costly.

Latest local model metrics:

- Precision: `0.459`
- Recall: `0.866`
- F1-score: `0.600`
- ROC-AUC: `0.839`
- Decision threshold: `0.28`

### CNN Image Classifier

The image classifier demonstrates how a CNN learns visual patterns directly from pixels.

Key ideas:

- Input images are converted to RGB, center-cropped/resized to `32x32`, and normalized.
- Convolution layers learn image features.
- MaxPooling reduces spatial size.
- Dropout helps control overfitting.
- Softmax returns probabilities for 10 classes.

The full training script uses CIFAR-10:

```powershell
python train_cnn.py
```

If CIFAR-10 download is slow or unavailable, a quick local demo model can be generated with:

```powershell
python bootstrap_cnn_demo.py
```

## Run Locally

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the combined app:

```powershell
python -m streamlit run app.py
```

Or run each project separately:

```powershell
python -m streamlit run pages\1_Churn_Predictor.py --server.port 8502
python -m streamlit run pages\2_Image_Classifier.py --server.port 8503
```

## Training

Train the churn model:

```powershell
python train_churn.py
```

Train the CNN on CIFAR-10:

```powershell
python train_cnn.py
```

## Presentation Summary

These projects show two common AI workflows:

- Classical machine learning for structured tabular data.
- Deep learning for image data.

The churn project focuses on business cost, class imbalance, and leak-free evaluation. The image project focuses on feature learning from pixels using convolutional neural networks.
