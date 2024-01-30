
---

# Predictive Maintenance using Machine Learning

## Overview

This repository contains the code for predictive maintenance using machine learning, focusing on predicting potential failures in machinery based on historical data and sensor information.

## Requirements

- **Python 3.x**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Keras**
- **Matplotlib**

Install dependencies using:

```bash
pip install pandas numpy scikit-learn keras matplotlib
```

## Usage

1. Import the necessary libraries in your Python script:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

plt.style.use('ggplot')  # Use 'ggplot' style for matplotlib plots
```



## Results

-**Balanced Accuracy Score: 0.5688**
-**Accuracy Score: 0.9518**
-**Matthews Correlation Coefficient: 0.4955**

## License

This project is licensed under the [MIT License](LICENSE).

---
