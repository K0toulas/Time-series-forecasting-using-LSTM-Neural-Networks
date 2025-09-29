# Time series forecasting using LSTM-Neural-Networks 

This project implements and evaluates a three-layer **Long Short-Term Memory (LSTM)** neural network to predict the minute-by-minute closing price of Bitcoin. It compares the performance of the advanced LSTM model against a traditional time-series forecasting method, **ARIMA**, to demonstrate the superior capability of deep learning in handling highly volatile and non-linear cryptocurrency data.


---

##  Project Objectives

The primary objective was to design, train, and evaluate a three-layer LSTM model for accurate price prediction, using a dataset of minute-by-minute Bitcoin prices from January 1, 2021, to March 1, 2022.

* **Model Implementation:** Implement a three-layer LSTM network for Bitcoin price prediction.
* **Performance Comparison:** Compare the LSTM model's performance against a baseline ARIMA model.
* **Evaluation:** Evaluate model accuracy by predicting Bitcoin prices for the last ten days of February 2022.
* **Efficiency Analysis:** Analyze the computational efficiency of the model in terms of training and inference (forecasting) times.

---

##  Model Architecture & Methodology

The core of the project is a three-layer LSTM network, chosen after research indicated multi-layer LSTM architectures were effective in cryptocurrency prediction.

### Network Structure

The architecture consists of an input layer, two LSTM hidden layers, and a final dense output layer.

| Layer Type | Units/Output | Activation Function | Purpose |
| :--- | :--- | :--- | :--- |
| **Input Layer** | `sequence_length` (Normalized with MinMax) | - | Receives the historical minute-by-minute closing prices. |
| **LSTM Layer 1** | 50 cells | ReLU | Feature extraction and long-term dependency capture. |
| **LSTM Layer 2** | 50 cells | ReLU | Further processing of sequential features. |
| **Dense Layer (Output)** | 1 | Linear | Produces the single predicted closing price. |

### Data and Training Details

| Parameter | Value in `project.py` (Model 1) | Description |
| :--- | :--- | :--- |
| **Dataset** | `BTC-2021_01_01_2022_03_01_min_basis.csv` | Minute-by-minute Bitcoin prices used for training. |
| **Sequence Length** | **1440** | The amount of past data (minutes/one day) the model looks at to make a prediction. |
| **Epochs** | **10** | The number of training cycles over the full dataset. |
| **Batch Size** | **32** | The number of samples processed in a forward and backward pass. |
| **Loss Function** | **MSE** (Mean Square Error) | The metric used for training and validation. |
| **Optimizer** | **Adam** | The chosen optimization algorithm. |
| **ARIMA Order** | **(5, 1, 1)** | The parameters chosen for the ARIMA baseline model. |

### Forecasting vs. Validation

| Phase | Prediction Mechanism | Input/Process |
| :--- | :--- | :--- |
| **Validation** | Predictions made on known historic data. | Model makes predictions for each minute of the period but **does not use its previous prediction** to make the next one. |
| **Forecasting** | Predictions made for future prices without knowledge of actual values. | After each prediction, the sequence is rolled and the oldest price is replaced with the newly predicted price (**error is compounded**). |

---

##  How to Run the Project

The entire model, including data preprocessing, training, validation, forecasting, and plotting, is contained within `project.py`.

### Prerequisites

You need **Python 3** and the following libraries. Install them using `pip`:

    ```bash
    pip install numpy pandas tensorflow scikit-learn matplotlib statsmodels

### Execution Steps 
1. Data: Ensure your minute-by-minute Bitcoin price data file (BTC-2021_01_01_2022_03_01_min_basis.csv) is in the same directory as project.py.
2. Run: Execute the main script:
   
        ```bash
        python project.py
        **NOTE:** The training process, especially for Model 1 (35 hours), is computationally intensive and will **take a long time to complete** (approximately 13-35 hours depending on your chosen parameters and hardware).
   
The script will perform the following actions:
* Load and normalize the data.
* Build and compile the LSTM model.
* Train the model for epoch iterations (default 10).
* Save the model weights to lstm_weights.txt.
* Display the training MSE plot.
* Perform validation on the last 10 days of historic data, calculate Validation MSE, and display the validation plot.
* Perform the 10-day forecast for both the LSTM and ARIMA models, displaying the final comparative plot.
* ***take a long time to complete***

---
### Results Summary
The LSTM model generally performed better than the ARIMA model, which essentially flat-lined its prediction due to Bitcoin's volatility. The forecasting results, however, were not satisfactory, suffering from an oscillating effect.

### Comparison of the Two Models 
| Metric | Model 1 (Seq Len: 1440 min) | Model 2 (Seq Len: 420 min) | Notes |
| :--- | :---: | :---: | :--- |
| **Epochs** | 6 | 10 | |
| **Training Time** | 35 hours | 13.5 hours | Larger sequence length leads to greater computational overhead. |
| **Final Training MSE** | $2.9035 \times 10^{-6}$ | $2.7852 \times 10^{-6}$ | Training MSE is very low. |
| **Final Validation MSE** | **3183.69** | **2058.89** | The larger sequence length model (Model 1) was less accurate in short-term validation. |
| **Forecast Elapsed Time (10 days)** | 1161 seconds | 1052 seconds | The time to complete the 10-day forecast is around 20 minutes for both. |

Forecasting Challenge:
  * Both LSTM models demonstrated an oscillating effect in the forecast, a common problem with LSTM networks due to the feedback of their own predictions (compounded error).
  * The ARIMA model's prediction essentially flat-lined due to its struggle to handle Bitcoin's extreme volatility.

### Future work and limitations
**Identified Limitations**
* Computational Constraints: The LSTM model demanded substantial computational resources and training time (up to 35 hours). This limitation significantly restricted the extent of hyperparameter optimization and experimentation with larger sequence lengths and more complex multi-layered architectures.
* Exclusion of External Factors: The model relied solely on historical price data. A major limitation was the exclusion of crucial exogenous variables, such as social media sentiment and macroeconomic indicators, which are known drivers of Bitcoin's volatile price movements.

**Future Directions**
* Advanced Model Architectures: Future work should explore more sophisticated models designed for complex sequence data. This includes investigating hybrid models (e.g., combining ARIMA and LSTM for feature capture) or adopting cutting-edge deep-learning frameworks like Transformer models to potentially enhance long-term accuracy and training efficiency.
* Data Granularity Optimization: Given the difficulty in achieving stable long-term forecasts with minute-by-minute data, an exploration into using daily closing prices is warranted. This change in data granularity could yield better results for longer-horizon predictions.
* Feature Engineering: The most critical step involves incorporating external data sources. Integrating cleaned and processed sentiment scores, news volume, and relevant economic indicators as additional features could significantly improve the model's ability to capture non-linear price influences.


