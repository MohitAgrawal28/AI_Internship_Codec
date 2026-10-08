# Live Demo Script (≈ 8–10 minutes)

> Fill the `[ ]` placeholders with the numbers shown on the **Model performance** tabs of your own deployed app. Don't quote numbers you haven't seen on screen.

---

## 0. Opening (30 sec)

"Good morning. During my 8-week AI internship at Codec Technologies I built two end-to-end systems that cover both halves of applied AI: **classical machine learning on tabular data**, and **deep learning on images**. Instead of slides, I'll show both running live."

*(Open the home page.)*

---

## 1. Project 1: Churn Predictor (≈ 4–5 min)

### The problem (30 sec)
"Winning a new customer costs far more than keeping one. So the goal is to rank customers by churn risk early enough for a retention team to act. The key constraint: **churners are a minority**, so a model that always says 'will stay' looks highly accurate but is useless. That is why I never used accuracy as the headline metric."

### The workflow (60 sec) — *open the "How it works" tab*
1. **Audit & clean**: blank `TotalCharges` strings coerced to numbers, imputed *inside* the pipeline so statistics come from training data only.
2. **Feature engineering**: tenure buckets, average spend per month, count of add-on services.
3. **SMOTE** to rebalance the minority class, which creates *synthetic* churners by interpolating between real neighbours rather than duplicating rows.
4. **Random Forest**, tuned with randomised search over stratified 5-fold CV, optimising **F1 on the churn class**.
5. **Threshold tuning**: I moved the cut-off from 0.5 to **[threshold]** to favour recall, because missing a churner costs more than a wasted discount.

### The one decision I'm proudest of (30 sec)
"SMOTE sits **inside** the cross-validation pipeline. My first version oversampled before splitting, and the scores looked fantastic, but they were inflated by leakage. Moving SMOTE into an `imblearn` pipeline means synthetic points are created only from training folds, and validation/test data keep their real class ratio."

### Live demo (90 sec)
1. Select **"High-risk newcomer"** → *Predict*. "Two months tenure, month-to-month contract, electronic check, fibre, no add-ons → probability **[x]%**, flagged, with a retention action suggested."
2. Select **"Loyal long-term customer"** → *Predict*. "Two-year contract, five years tenure, add-ons → low risk."
3. Change one slider (e.g. contract to *Two year* on the first profile) → "Watch how much the risk drops. This is the lever a retention team would pull."

### Results & impact (45 sec) — *open "Model performance"*
"On the untouched test set: precision **[ ]**, recall **[ ]**, F1 **[ ]**, ROC-AUC **[ ]**. The confusion matrix shows exactly where it errs. The top drivers are **contract type, tenure and monthly charges**, which match what I saw in exploratory analysis, so the model is credible to a business audience, not just numerically good."

---

## 2. Project 2: CNN Image Classifier (≈ 3–4 min)

### The problem (20 sec)
"Classical models need hand-crafted features, which is impractical for images. A CNN learns edges → textures → object parts directly from pixels."

### Architecture (60 sec) — *open the "Architecture" tab*
- Input: **32×32×3** tensors, normalised to [0, 1].
- Two **Conv2D → MaxPool → Dropout** blocks (32 then 64 filters). Convolutions share weights, so a feature learned in one place is detected everywhere; pooling halves the spatial size and adds tolerance to small shifts.
- **Dense 128 → Dropout 0.5 → Softmax** head gives a probability for each class.
- Trained with **Adam** and **categorical cross-entropy**, with **EarlyStopping** and learning-rate reduction on plateau. **Dropout and augmentation** were added because my first version without them memorised the training set.

### Live demo (90 sec)
1. Choose **"Use a sample test image"** → pick *cat_1* → show prediction, confidence, and the **"what the model sees"** 32×32 view. "That is all the information it gets."
2. Pick a *truck* or *dog* sample → "Notice the second-highest class. Automobile/truck and cat/dog are the visually similar pairs."
3. **Upload** a clear photo of a car/ship/plane → show the Softmax distribution.
4. *(Optional, shows honesty)* Upload something outside the classes → low-confidence warning. "A closed-set classifier must pick one of its 10 classes, so confidence is the signal to watch."

### Results (30 sec) — *open "Training & metrics"*
"Test accuracy **[ ]**. The train/validation curves show **[small / moderate]** gap, so overfitting is under control. Per-class metrics and the confusion matrix show where it struggles: **[classes]**."

---

## 3. Close (45 sec)

"Both projects share one principle: **a model is only trustworthy if the evaluation is.** That meant leak-free pipelines, metrics that match business cost, a held-out test set touched once, and fixed seeds with pinned dependencies for reproducibility.

**Next steps:** for churn, gradient-boosted trees, probability calibration and SHAP explanations, then a FastAPI service with drift monitoring. For vision, transfer learning with MobileNet/ResNet and TensorFlow Lite for edge deployment, which fits my electronics background.

Thank you. I'm happy to take questions."

---

## Likely questions: short answers

| Question | Answer |
|---|---|
| Why not just report accuracy? | With a minority churn class, "always predict stay" scores high accuracy while catching zero churners. Recall/F1/PR-AUC reflect actual business cost. |
| Why SMOTE and not simple duplication? | Duplicates cause overfitting to exact rows. SMOTE interpolates new plausible points. It is applied only to training folds to avoid leakage. |
| Why Random Forest? | It captures non-linear interactions (e.g. month-to-month × high charges), needs little scaling, is robust to variance, and gives feature importances. A Logistic Regression baseline was kept for comparison. |
| Why threshold ≠ 0.5? | The cost of a missed churner exceeds the cost of a wasted offer, so I traded some precision for recall using the precision–recall curve. |
| Why is CNN accuracy only [ ]%? | It is a compact network trained from scratch on 32×32 images. Transfer learning with pretrained backbones is the planned improvement. |
| Why does my own photo sometimes fail? | The model sees only 32×32 pixels and only knows 10 classes. Photos with cluttered backgrounds or other objects fall outside its training distribution. |
| How would you deploy this for real? | Containerise with Docker, serve via FastAPI, track experiments with MLflow, and monitor data drift. |
| How do you know it isn't overfitting? | CV scores, a separate validation set, train/validation curves, Dropout, EarlyStopping, and a test set evaluated once. |

---

## Pre-demo checklist
- [ ] Open both pages 10 minutes early (free Streamlit apps sleep; the first load of TensorFlow takes a few seconds).
- [ ] Run each preset once to warm up the model.
- [ ] Have the sample images ready in case the upload fails over Wi-Fi.
- [ ] Keep a local `streamlit run app.py` as a backup.
