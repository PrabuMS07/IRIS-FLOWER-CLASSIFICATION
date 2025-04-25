

# 🌸 Iris Flower Classification 🐍📊

This project uses a Random Forest Classifier, implemented in Python with Scikit-learn, to classify Iris flower species (`setosa`, `versicolor`, `virginica`) based on their sepal and petal measurements.

## 📝 Project Overview

This repository contains a Python script (`iris_classifier.py` - *please rename appropriately if needed*) that performs the following steps:

1.  **Loads Data:** Reads the classic `IRIS.csv` dataset (included in the repository).
2.  **Data Preparation:** Separates the features (sepal length/width, petal length/width) from the target variable (species).
3.  **Train-Test Split:** Divides the data into training (80%) and testing (20%) sets (`random_state=42` for reproducibility).
4.  **Feature Scaling:** Applies `StandardScaler` to standardize the feature values, ensuring they have zero mean and unit variance. This is often beneficial for many machine learning algorithms.
5.  **Model Training:** Initializes and trains a `RandomForestClassifier` (with 100 trees and `random_state=42`) using the scaled training data.
6.  **Prediction:** Uses the trained model to predict the species for the scaled test data.
7.  **Evaluation:**
    *   Prints the **Confusion Matrix** to the console, showing correct and incorrect predictions for each class.
    *   Prints the **Classification Report**, detailing precision, recall, f1-score, and support for each species, along with overall accuracy.
8.  **Visualization:** Generates and displays a heatmap visualization of the Confusion Matrix using Matplotlib and Seaborn for a clearer visual interpretation of the model's performance on the test set.

## 💾 Dataset

*   **File:** `IRIS.csv` (Included in this repository)
*   **Source:** The classic Iris flower dataset, a staple in machine learning tutorials.
*   **Columns:**
    *   `sepal_length`: Sepal length in cm.
    *   `sepal_width`: Sepal width in cm.
    *   `petal_length`: Petal length in cm.
    *   `petal_width`: Petal width in cm.
    *   `species`: The species of the Iris flower (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`).

## ✨ Features & Target

*   **Features (X):** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
*   **Target Variable (y):** `species`

## ⚙️ Technologies & Libraries

*   Python 3.x
*   Pandas
*   Scikit-learn (`train_test_split`, `StandardScaler`, `RandomForestClassifier`, `classification_report`, `confusion_matrix`)
*   Matplotlib
*   Seaborn

## 🛠️ Setup & Installation (using VS Code)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PrabuMS07/IRIS-FLOWER-CLASSIFICATION.git
    cd IRIS-FLOWER-CLASSIFICATION
    ```
2.  **Open Folder:** Open the cloned folder in Visual Studio Code (`File` > `Open Folder...`).
3.  **Python Interpreter:** Ensure you have a Python 3 interpreter selected in VS Code.
4.  **Terminal:** Open the integrated terminal in VS Code (`View` > `Terminal` or `Ctrl + \``).
5.  **(Optional but Recommended) Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1 or cmd: venv\Scripts\activate.bat
    ```
6.  **Install Dependencies:** Create a `requirements.txt` file in the folder with this content:
    ```txt
    pandas
    scikit-learn
    matplotlib
    seaborn
    ```
    Then run the installation command in the terminal:
    ```bash
    pip install -r requirements.txt
    ```
7.  **Dataset:** The `IRIS.csv` file is already included in the repository.

## ▶️ Usage (within VS Code)

1.  Make sure you have completed the Setup steps (and activated the virtual environment if created).
2.  Open your Python script file (e.g., `iris_classifier.py` - **you'll need to save the script code you provided into a `.py` file**) in the VS Code editor.
3.  Run the script from the VS Code terminal:
    ```bash
    python your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual name of your Python file)*

4.  **Output:**
    *   The first 5 rows of the dataset (`data.head()`) will be printed to the terminal.
    *   The Confusion Matrix (as text) will be printed to the terminal.
    *   The Classification Report (Precision, Recall, F1-Score, Accuracy) will be printed to the terminal.
    *   A **new window** will pop up displaying the **Confusion Matrix Heatmap** visualization. Close this window to allow the script to finish completely if needed.

## 📊 Interpreting the Results

*   **Confusion Matrix:** Shows how many instances of each true class were predicted as each possible class. The diagonal elements represent correctly classified instances. Off-diagonal elements represent misclassifications.
*   **Classification Report:**
    *   **Precision:** Out of all instances predicted as a certain species, what proportion were actually that species? (TP / (TP + FP))
    *   **Recall (Sensitivity):** Out of all instances that were actually a certain species, what proportion did the model correctly predict? (TP / (TP + FN))
    *   **F1-Score:** The harmonic mean of Precision and Recall, providing a single metric balancing both. (2 * (Precision * Recall) / (Precision + Recall))
    *   **Accuracy:** The overall proportion of correctly classified instances. (Total Correct / Total Instances)
*   **Confusion Matrix Heatmap:** A visual representation of the confusion matrix, making it easier to spot patterns of misclassification (if any).

## 💡 Potential Future Improvements

*   **Hyperparameter Tuning:** Use techniques like `GridSearchCV` or `RandomizedSearchCV` to find the optimal `n_estimators`, `max_depth`, etc., for the `RandomForestClassifier`.
*   **Cross-Validation:** Implement k-fold cross-validation during training/evaluation for a more robust assessment of model performance.
*   **Explore Other Models:** Compare the performance of Random Forest with other classification algorithms like Support Vector Machines (SVM), Logistic Regression, K-Nearest Neighbors (KNN), or Gradient Boosting.
*   **Feature Importance:** Analyze and visualize the feature importances provided by the trained Random Forest model (`model.feature_importances_`) to understand which measurements contribute most to the classification.

---
