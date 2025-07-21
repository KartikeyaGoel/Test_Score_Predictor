# SOL Score Predictor

This project uses educational, demographic, and behavioral data to predict the **SOL (Standards of Learning) pass rate** for Virginia public schools using a **Random Forest Regressor**. It loads data from multiple sources, preprocesses and merges them, and trains a regression model to forecast performance based on school-level factors.

---

## 📦 Requirements

- Python 3.8+
- Required libraries:

```bash
pip install pandas numpy scikit-learn openpyxl
```

> `tkinter` is used for optional GUI-based input selection. It comes pre-installed with most Python distributions.

---

## 🚀 How to Run

1. **Ensure the following folder structure is set up (relative to the script):**

```
archive/
├── Demographics/
│   ├── Economically_Disadvantaged.csv
│   ├── English_Learners.csv
│   ├── Foster_Care.csv
│   └── ... (other demographic files)
├── Economic Factors/
│   ├── Free_and_Reduced_Lunch.csv
│   └── Funding.csv
├── Student Behaviors/
│   ├── Absenteeism.csv
│   ├── Dropout.csv
│   └── Graduation.csv
├── Teachers/
│   ├── Education_Level.csv
│   ├── Experience.csv
│   └── Licensure.csv
└── Testing/
    ├── SOL_Pass_Rate.csv
    └── school-by-subject-2022.xlsx
```

2. **Run the script:**

```bash
python your_script.py
```

This will:
- Load and preprocess all datasets
- Perform feature engineering
- Merge into a unified dataset
- Train a `RandomForestRegressor` with hyperparameter tuning
- Save the merged data to `Factors_Used.csv`
- Print the predicted SOL pass rate based on either manual or file-based input

---

## 📝 Input Methods

You can either:
- Provide a sample input text file (`Sample.txt`) for prediction
- Or use the built-in Tkinter window to select files manually

> The script maps feature keys from the input file to relevant datasets automatically.

---

## 📤 Output

- `Factors_Used.csv`: Final dataset used for model training.
- Console: Predicted SOL pass rate and MSE from test evaluation.

---

## 📁 Example Input Format (Sample.txt)

```txt
disa_choice1_percent_x: 0.25
engl_choice1_percent: 0.13
Black, not of Hispanic origin: 0.12
...
sol pass rate: 78.5
```

Make sure keys match the expected feature names in the code.

---

## 🛠 Notes

- The model is trained on 80% of the data and tested on 20% using `train_test_split`.
- Input data is scaled using `StandardScaler`.
- Prediction is based on cleaned and structured multi-source educational data.

---

Feel free to customize the model, add UI enhancements, or export results to different formats.
