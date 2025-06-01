# Music Analysis Project

A comprehensive music analysis framework that combines traditional audio features with advanced entropy-complexity measures and fractal dimensions for genre classification and music understanding.

## 🎵 Overview

This project analyzes music using multiple approaches:
- **Traditional Spotify Features**: Standard audio features like danceability, energy, loudness, etc.
- **Entropy-Complexity Analysis**: Permutation entropy and complexity measures across different audio domains
- **Fractal Dimensions**: Box-counting and Higuchi fractal dimensions for complexity quantification
- **Machine Learning Classification**: Advanced genre classification using ensemble methods

## 🚀 Features

### Core Analysis Methods
- **Amplitude Analysis**: Hilbert envelope-based entropy-complexity measures
- **Spectral Flux Analysis**: STFT-based temporal spectral changes
- **Harmonic Analysis**: CQT chroma-based harmonic complexity
- **Spectral Entropy**: Frame-wise spectral entropy analysis
- **Fractal Dimensions**: Box-counting and Higuchi methods for complexity quantification

### Machine Learning
- **Multi-class Genre Classification**: Support for multiple music genres
- **Feature Importance Analysis**: Understanding which features drive classification
- **Ensemble Methods**: Random Forest, Gradient Boosting, SVM, and Logistic Regression
- **Hyperparameter Optimization**: Automated tuning using RandomizedSearchCV and HalvingGridSearchCV
- **Model Calibration**: Probability calibration for better predictions

### Visualization
- **Entropy-Complexity Space**: 2D plots showing the relationship between entropy and complexity
- **Feature Distributions**: Genre-wise feature distribution analysis
- **Correlation Analysis**: Pearson and Spearman correlation matrices
- **Confusion Matrices**: Classification performance visualization
- **Fractal Dimension Plots**: Box-counting and Higuchi FD visualizations

## 📁 Project Structure

```
musicAnalysis/
├── METHODS.py                          # Core analysis methods
├── genre_classification_model.py       # Model training and evaluation
├── binary_data_analysis.py             # Binary classification analysis
├── jsd_binary_genre_analysis.py        # Jensen-Shannon divergence analysis
├── correlation_pearson_spotify.py      # Pearson correlation analysis
├── correlation_spearman_spotify.py     # Spearman correlation analysis
├── scripts/                            # Data processing scripts
│   ├── extract_features.py             # Feature extraction pipeline
│   ├── spotify_script.py               # Spotify API integration
│   ├── data_download_PE_C.py           # Data download utilities
│   └── ...
├── visual/                             # Visualization modules
│   ├── BOX_VISUAL.py                   # Box-counting visualization
│   ├── HIGUICHI_VISUAL.py              # Higuchi FD visualization
│   ├── CHROMA.py                       # Chroma visualization
│   ├── FLUX.py                         # Spectral flux visualization
│   └── ...
├── plots/                              # Generated plots and visualizations
│   ├── binary/                         # Binary analysis results
│   ├── jsd/                            # JSD analysis results
│   ├── spearman/                       # Spearman correlation results
│   └── sampled_dataset_PE_C/           # Main analysis results
├── dataforGithub/                      # Processed datasets
│   └── csv/
│       ├── sampled_dataset_PE_C_fd.csv # Main dataset with all features
│       ├── balanced_dataset.csv        # Balanced genre dataset
│       └── ...
├── notebooks/                          # Jupyter notebooks for analysis
└── requirements.txt                    # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd musicAnalysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Quick Start - Genre Classification

```python
# Run the genre classification pipeline
python genre_classification_model.py
```

This will:
- Load the pre-processed dataset
- Perform stratified train-test split
- Train multiple models with hyperparameter optimization
- Generate classification reports and confusion matrices
- Create feature importance visualizations

### Advanced Analysis

```python
# Perform correlation analysis
python correlation_pearson_spotify.py
python correlation_spearman_spotify.py

# Binary genre analysis
python binary_data_analysis.py

# Jensen-Shannon divergence analysis
python jsd_binary_genre_analysis.py
```

### Feature Extraction

```python
# Extract features from audio files
python scripts/extract_features.py

# Download data from Spotify
python scripts/spotify_script.py
```

### Visualization

```python
# Generate visualizations
python visual/BOX_VISUAL.py
python visual/HIGUICHI_VISUAL.py
python visual/CHROMA.py
python visual/FLUX.py
```

## 🔬 Analysis Methods

### 1. Entropy-Complexity Analysis

The project implements permutation entropy (PE) and complexity (C) measures across different audio domains:

- **Amplitude Domain**: Hilbert envelope analysis
- **Spectral Domain**: STFT-based spectral flux
- **Harmonic Domain**: CQT chroma analysis
- **Spectral Entropy**: Frame-wise entropy measures

### 2. Fractal Dimensions

- **Box-Counting FD**: 2D spectrogram-based fractal dimension
- **Higuchi FD**: 1D time series fractal dimension

### 3. Traditional Features

Standard Spotify audio features including:
- Danceability, Energy, Loudness
- Acousticness, Instrumentalness
- Speechiness, Liveness, Valence
- Tempo, Key, Mode, Time Signature

## 📈 Results

The project generates comprehensive results including:

- **Classification Performance**: Accuracy, precision, recall, F1-score
- **Feature Importance**: Which features contribute most to classification
- **Correlation Analysis**: Relationships between different features
- **Visualizations**: Entropy-complexity plots, confusion matrices, feature distributions
- **Statistical Analysis**: Effect sizes, p-values, normality tests

## 🎯 Supported Genres

The analysis supports multiple music genres including:
- Acoustic, Blues, Classical
- Country, Electronic, Folk
- Hip-Hop, Jazz, Metal
- Pop, Reggae, Rock
- And more...

## 📝 Dependencies

Key dependencies include:
- `librosa` - Audio processing
- `scikit-learn` - Machine learning
- `numpy`, `pandas` - Data manipulation
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `antropy`, `ordpy` - Entropy analysis
- `scipy` - Scientific computing
- `soundfile` - Audio file handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Spotify Web API for audio features
- Librosa for audio processing capabilities
- Scikit-learn for machine learning tools
- The open-source community for various analysis libraries

## 📞 Contact

For questions or contributions, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project requires audio files for feature extraction. Make sure you have the necessary permissions and licenses for any audio content you analyze.
