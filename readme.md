# Customer Churn Prediction with an ANN

Predicts whether a telecom customer will leave, using an Artificial Neural Network (Keras/TensorFlow), with a Streamlit app for live predictions.

## Dataset
Telco Customer Churn (Kaggle, IBM sample data): 7,043 customers, 20 features. About 27% of customers churned, so the classes are imbalanced.

## Approach
1. Explored the data and cleaned it (fixed `TotalCharges`, merged "No internet service" into "No").
2. Encoded categories (Yes/No to 1/0, one-hot for multi-category columns) and scaled numeric columns, fitting the scaler on training data only.
3. Built a logistic regression baseline.
4. Built an ANN: 23 inputs, Dense(16, ReLU), Dropout(0.3), Dense(8, ReLU), Dropout(0.2), Dense(1, sigmoid), with early stopping.
5. Improved churn recall with a lower decision threshold and with class weights.

## Results (test set, churned class)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.803 | 0.648 | 0.567 | 0.605 |
| ANN (threshold 0.5) | 0.798 | 0.646 | 0.532 | 0.584 |
| ANN (threshold 0.25) | 0.726 | 0.490 | 0.805 | 0.609 |
| **ANN (class weights)** | 0.753 | 0.523 | 0.805 | **0.634** |

**Key findings**
- With default settings, the ANN performed about the same as logistic regression. On small tabular data with mostly simple patterns, that is common.
- Both models missed almost half of the churners. Class weights raised churn recall from 0.53 to 0.81 and gave the best F1.
- The trade-off is lower accuracy and precision (more false alarms). For a telecom company, a retention offer sent to a customer who would have stayed usually costs less than losing one who was leaving.
- These are results from one train/test split and one random seed, so small differences may shift between runs.

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app shows a churn probability with three risk levels: low (under 30%), medium (30% to 50%), and high (above 50%). Total charges is estimated as tenure × monthly charges.

## Project structure