# 🎮 PUBG Finish Placement Prediction (End-to-End ML Project)

## 📍 Project Overview
This project aims to predict the final ranking (**WinPlacePerc**) of a player in a **PUBG (PlayerUnknown's Battlegrounds)** match using Machine Learning. By analyzing in-game statistics such as kills, movement distance, and support items, we developed a model to estimate the winning probability.

This project was developed as part of the **MultiGroup / Zero2End Machine Learning Bootcamp**.

## ❓ Problem Definition
In Battle Royale games, survival is key. However, does "**killing more enemies**" guarantee a win, or is "**staying hidden**" a better strategy?
* **Goal:** Predict `winPlacePerc` (0-1) for a given player.
* **Dataset:** PUBG Finish Placement Prediction (Kaggle).
* **Metric:** Mean Absolute Error (MAE).

## 🚀 Model Performance
We compared a baseline model against an optimized model with engineered features.

| Model Stage | MAE Score | Improvement |
| :--- | :--- | :--- |
| **Baseline (LightGBM)** | `0.0637` | - |
| **Feature Engineering** | `0.0589` | +7.5% |
| **Final Model (Optimized)** | **`0.0552`** | **+13.3%** |

*The final model was optimized using `RandomizedSearchCV`.*

## 🛠️ Feature Engineering Strategy
To improve the model, several new features were derived:
1.  **Total Distance:** Sum of `walkDistance`, `rideDistance`, and `swimDistance`.
2.  **Health Items:** Sum of `heals` and `boosts`.
3.  **Relative Performance:** Player's kills and damage compared to the **match average** (Since every match has a different difficulty level).
4.  **Teamwork:** Sum of `revives` and `assists`.

## 💻 Tech Stack
* **Language:** Python 3.9+
* **ML Library:** LightGBM, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Web App:** Streamlit
* **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
```text
PUBG_Win_Prediction/
├── data/                  # Raw and processed data (Not included in repo)
├── models/                # Trained model files (.pkl)
├── notebooks/             # Jupyter Notebooks for analysis
│   ├── 1_EDA.ipynb
│   ├── 2_Baseline.ipynb
│   ├── 3_Feature_Engineering.ipynb
│   └── 4_Model_Optimization.ipynb
├── src/                   # Source code
│   ├── inference.py       # Prediction logic
├── banner.jpg
├── app.py                 # Streamlit Frontend application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

# ⚙️ How to Run Locally
1.  **Download the project:**
    Clone this repository or download the files.

2.  **Download the Dataset:**
    The dataset is too large to be included in the repository.
    * Go to: [PUBG Finish Placement Prediction (Kaggle)](https://www.kaggle.com/c/pubg-finish-placement-prediction/data)
    * Download `train_V2.csv`.
    * Create a folder named `data/raw/` inside the project directory.
    * Place the `train_V2.csv` file into `data/raw/`.
3. Create and Activate Virtual Environment: Open a terminal in the project folder and run the following commands to create a clean enviroment:
  - Windows:
    ```bash
    python -m venv env
    venv\Scripts\activate
    ```
  - macOS/Linux:
    ```bash
    python3 -m venv env
    source venv/bin/activate
    ```
4. Install dependencies: Once the enviroment is active, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the App:
   ```bash
   streamlit run app.py
   ```
