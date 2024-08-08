# Stock Price Prediction using LSTM


## Introduction
Accurate stock price prediction is a longstanding challenge in the field of finance and economics. Traditional statistical models often struggle to capture the inherent complexities and non-linear relationships present in stock market data. The emergence of deep learning techniques, particularly the LSTM architecture, has shown promising results in addressing this challenge.

## Methodology
The stock price prediction model follows these main steps:

1. **Data Preprocessing**:
   - Obtained the historical stock data for Microsoft (MSFT) from 2002 to 2024 using the `yfinance` library.
   - Preprocessed the data by handling missing values, removing duplicates, and converting the data to a suitable format for the model.
   - Performed feature scaling using the `MinMaxScaler` to normalize the data between 0 and 1, ensuring that all features have a similar range and preventing the domination of features with larger ranges.

2. **Train-Test Split**:
   - Divided the dataset into training (80%) and testing (20%) sets to evaluate the model's performance.

3. **LSTM Model Architecture**:
   - Designed a sequential LSTM model with the following layers:
     - LSTM layer with 50 units, `return_sequences=True`, and `input_shape=(X_train.shape[1], 1)`.
     - LSTM layer with 50 units and `return_sequences=False`.
     - Dense layer with 25 units.
     - Dense layer with 1 unit (for the single output, which is the predicted stock price).
   - Compiled the model with the Adam optimizer and Mean Squared Error (MSE) loss function.

4. **Model Training and Evaluation**:
   - Trained the model for 75 epochs with a batch size of 32.
   - Evaluated the model's performance on the test set using the Root Mean Squared Error (RMSE) metric.

5. **Model Deployment and Forecasting**:
   - Used the trained model to predict the next day's closing price for Microsoft stock.
   - Compared the predicted price with the actual closing price obtained from `yfinance`.

## AI Architecture
The LSTM model architecture used in this project is a type of Recurrent Neural Network (RNN) that is particularly well-suited for processing sequential data, such as time series data. The LSTM units in the model are designed to capture long-term dependencies in the stock price data, allowing the model to learn patterns and make more accurate predictions.

The specific architecture of the model is as follows:

1. **LSTM Layer 1**:
   - This layer has 50 units and `return_sequences=True`, which means it outputs a sequence of 50-dimensional vectors, one for each time step in the input.
   - The `input_shape=(X_train.shape[1], 1)` parameter specifies that the input to this layer should have a shape of (batch_size, 60, 1), where 60 is the number of past time steps used as input, and 1 is the number of features (in this case, the closing price).

2. **LSTM Layer 2**:
   - This layer also has 50 units, but `return_sequences=False`, which means it outputs a single 50-dimensional vector, representing the final state of the LSTM unit.

3. **Dense Layer 1**:
   - This is a fully connected layer with 25 units.
   - This layer helps the model learn more complex non-linear relationships in the data.

4. **Dense Layer 2**:
   - This is the final output layer with a single unit, representing the predicted stock price.

The choice of these layer configurations and hyperparameters was based on empirical experimentation and common practices in stock price prediction using LSTMs. The two LSTM layers allow the model to capture both short-term and long-term dependencies in the stock price data, while the dense layers help to further refine the predictions.

## Impact
The ability to accurately predict stock prices can have a significant impact on investment strategies, portfolio management, and financial decision-making. Successful stock price prediction models can help investors make more informed decisions, potentially leading to higher returns and reduced risk.

## Code Snippets and Explanation

### Feature Scaling and MinMaxScaler
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
```
Feature scaling is an important data preprocessing step in machine learning models, especially for neural networks. The `MinMaxScaler` is used to normalize the data between 0 and 1, which helps to prevent features with larger ranges from dominating the learning process. This is particularly important for stock price data, where the absolute values of the prices can vary significantly. By scaling the data, the model can focus on the relative changes in the stock price rather than the absolute values, leading to better generalization and faster convergence during training.

### LSTM Model Architecture
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
```
1. **LSTM Layer 1**:
   - The first LSTM layer has 50 units, which determines the dimensionality of the output vectors from this layer.
   - The `return_sequences=True` parameter tells the layer to output a sequence of 50-dimensional vectors, one for each time step in the input.
   - The `input_shape=(X_train.shape[1], 1)` parameter specifies that the input to this layer should have a shape of (batch_size, 60, 1), where 60 is the number of past time steps used as input, and 1 is the number of features (in this case, the closing price).

2. **LSTM Layer 2**:
   - The second LSTM layer also has 50 units, but `return_sequences=False`, which means it outputs a single 50-dimensional vector, representing the final state of the LSTM unit.
   - This layer helps the model capture longer-term dependencies in the stock price data.

3. **Dense Layer 1**:
   - This is a fully connected layer with 25 units.
   - This layer allows the model to learn more complex non-linear relationships in the data.

4. **Dense Layer 2**:
   - The final layer is a Dense layer with a single unit, representing the predicted stock price.

The choice of these layer configurations and hyperparameters was based on common practices in stock price prediction using LSTMs, as well as empirical experimentation to find the optimal architecture for the given problem and dataset.

### Model Compilation
```python
model.compile(optimizer='adam', loss='mean_squared_error')
```
The model was compiled with the Adam optimizer and Mean Squared Error (MSE) loss function. The Adam optimizer is a popular choice for training neural networks due to its adaptive learning rate and momentum-based updates, which can lead to faster convergence and better performance. The MSE loss function is commonly used for regression problems, as it penalizes large errors more heavily, which is desirable for accurate stock price prediction.

## Results
The trained LSTM model achieved a Root Mean Squared Error (RMSE) of 5.12 on the test set, which represents the average deviation of the predicted prices from the actual prices. This suggests that the model was able to capture the underlying patterns in the stock price data and make reasonably accurate predictions.

## Visualizations
### Training Loss Curve
![1](https://github.com/user-attachments/assets/17c12593-c3d5-4fa3-a72b-748bf74e1387)

The training loss curve shows the decrease in the mean squared error loss during the training process. The model was able to converge to a low loss value, indicating that it was able to learn the patterns in the stock price data effectively.

### Actual vs Predicted Prices
![2](https://github.com/user-attachments/assets/3a8d1583-ffc1-410d-ba0d-ecfffffd5c15)

The plot compares the actual closing prices (in blue) with the predicted closing prices (in orange) for the test set. The model's predictions generally follow the overall trend of the actual prices, demonstrating its ability to capture the dynamics of the stock market.

## Conclusion
The LSTM-based stock price prediction model developed in this project has shown promising results in forecasting the closing price of Microsoft stock. The model was able to effectively capture the complex non-linear relationships and long-term dependencies present in the stock price data, leading to reasonably accurate predictions.

## Challenges
One of the main challenges in this project was the inherent volatility and unpredictability of the stock market. Stock prices can be influenced by a wide range of factors, including economic conditions, geopolitical events, and investor sentiment, which can be difficult to accurately model using historical data alone. Additionally, the model's performance may be sensitive to the choice of hyperparameters and the architecture of the LSTM network.

## Future Improvements
To further improve the performance of the stock price prediction model, several enhancements could be considered:

1. **Incorporating Additional Features**: Incorporating other relevant features, such as macroeconomic indicators, industry-specific data, or news sentiment, could potentially improve the model's predictive power.

2. **Exploring Alternative Model Architectures**: Experimenting with different LSTM architectures, such as stacked LSTMs or bidirectional LSTMs, or using other types of neural networks, such as Convolutional Neural Networks (CNNs), may lead to better performance.

3. **Ensemble Modeling**: Combining the predictions of multiple models, such as LSTM, traditional time series models, and other machine learning algorithms, could potentially result in more robust and accurate predictions.

4. **Incorporating Uncertainty Quantification**: Developing methods to estimate the uncertainty associated with the model's predictions, such as using techniques like Bayesian neural networks or Monte Carlo dropout, could provide more informative and reliable forecasts.

5. **Improving Data Preprocessing**: Exploring alternative data preprocessing techniques, such as handling outliers, dealing with missing data, or incorporating domain-specific knowledge, may further enhance the model's performance.

6. **Expanding the Scope**: Extending the model to predict stock prices for a broader range of companies or industries could increase its practical utility and impact.

By addressing these areas for improvement, the stock price prediction model can be further refined and developed to provide more accurate and reliable forecasts, ultimately supporting more informed investment decisions and financial planning.
