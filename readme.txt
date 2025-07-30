# Employee Income Prediction using Machine Learning

This project uses a Random Forest Classifier to predict whether an individual earns more than \$50K per year based on features such as age, education, occupation, and other demographic data. The model is trained and evaluated using the [Adult Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult), also known as the "Census Income" dataset.

---

📊 Dataset

The dataset contains various features such as:

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Hours per week
- Native country
- Income (target variable)

---

## ⚙️ Project Workflow

1. **Data Cleaning**  
   - Removed missing or invalid entries  
   - Encoded categorical variables using `LabelEncoder`

2. **Feature Engineering**  
   - Dropped unnecessary columns (like `fnlwgt`)  
   - Scaled numerical features (optional step)

3. **Model Training**  
   - Trained using **Random Forest Classifier**  
   - Evaluated using accuracy, confusion matrix, and classification report

4. **Evaluation Metrics**
   - Accuracy score
   - Confusion matrix
   - Precision, Recall, F1-score

5. **Feature Importance Visualization**

---

## 🧪 How to Run

### 🔧 Prerequisites

Make sure you have Python 3.x and the required libraries installed.

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
