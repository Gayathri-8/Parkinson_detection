# Parkinson's Disease Classification Using Hybrid ML Model

## Project Overview
This project implements a hybrid machine learning model that combines Random Forest and Deep Learning (CNN-LSTM) approaches to classify Parkinson's disease stages using both audio features and clinical data.

## Features
- Audio feature extraction using librosa
- Clinical data preprocessing and feature engineering
- Hybrid model combining Random Forest and CNN-LSTM architecture
- Advanced data augmentation and balancing techniques
- Hyperparameter tuning using GridSearchCV

## Requirements
```python
numpy
pandas
librosa
tensorflow
scikit-learn
imbalanced-learn
seaborn
matplotlib
```

## Project Structure
```
├── prediction.py          # Main script containing model implementation
├── pva_wav_train.zip     # Training audio data
├── pva_wav_test.zip      # Testing audio data
├── plmpva_train.csv      # Training clinical data
└── plmpva_test-WithPDRS.csv  # Testing clinical data
```

## Data Processing
1. **Audio Processing**
   - Extracts MFCC features
   - Includes spectral features (centroids, rolloff)
   - Processes chroma and zero crossing rate
   - Combines multiple audio features for comprehensive analysis

2. **Clinical Data Processing**
   - Feature engineering with polynomial features
   - Handles missing values
   - Applies PCA for dimensionality reduction
   - Standardizes features using StandardScaler

## Model Architecture
### Random Forest
- Optimized using GridSearchCV
- Hyperparameter tuning for:
  - n_estimators
  - max_depth
  - min_samples_split
  - min_samples_leaf
  - max_features

### Deep Learning Model
- CNN layers for feature extraction
- Bidirectional LSTM layers for temporal patterns
- Dense layers with dropout and batch normalization
- Advanced regularization techniques

## Training Process
1. Data balancing using SMOTE
2. Class weight computation for imbalanced data
3. Implementation of early stopping
4. Learning rate reduction on plateau
5. Validation split for monitoring

## Model Evaluation
The model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

## Usage

### Running the Model
1. Ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

2. Place your data files in the project directory:
   - pva_wav_train.zip
   - pva_wav_test.zip
   - plmpva_train.csv
   - plmpva_test-WithPDRS.csv

3. Run the prediction script:
```bash
python prediction.py
```

### Streamlit Web Application
The project includes a web interface built with Streamlit for easy interaction with the model.

#### Requirements for Streamlit
```bash
pip install streamlit
```

#### Running the Streamlit App
1. Navigate to the project directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```

#### Features of the Web Interface
- Upload audio files for analysis
- Input clinical parameters through a user-friendly form
- Real-time prediction of Parkinson's disease stage
- Visualization of model confidence scores
- Display of feature importance

#### Using the Web Interface
1. Open the provided localhost URL in your web browser
2. Upload an audio file of the patient's voice
3. Fill in the required clinical parameters
4. Click "Predict" to get the classification result
5. View the detailed analysis and confidence scores

## Model Output
The script will output:
- Best RandomForest parameters
- Training progress for the deep learning model
- Final hybrid model performance metrics
