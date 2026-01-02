# Model Artifacts

This folder contains the trained machine learning models and vectorizers.

## Files

The following files are generated when you run `sentiment_analysis.ipynb`:

### Model Files (Not in Git - Too Large)
- `best_model.pkl` - Best performing model (Random Forest) - **168.91 MB**
- `random_forest_model.pkl` - Random Forest classifier
- `logistic_regression_model.pkl` - Logistic Regression classifier
- `linear_svm_model.pkl` - Linear SVM classifier
- `naive_bayes_model.pkl` - Naive Bayes classifier
- `gradient_boosting_model.pkl` - Gradient Boosting classifier
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer for feature extraction

### Metadata Files (In Git)
- `model_metadata.json` - Model performance metrics and metadata

## How to Generate These Files

1. Open and run the Jupyter notebook:
```bash
jupyter notebook sentiment_analysis.ipynb
```

2. Run all cells (`Cell > Run All`)

3. The models will be automatically trained and saved to this folder

## File Sizes

The model files are excluded from git because they exceed GitHub's 100MB file size limit:
- Total size: ~175 MB
- Largest file: `best_model.pkl` (168.91 MB)

## Using the Models

After generating the models, you can load them like this:

```python
import pickle

# Load the best model
with open('artifacts/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open('artifacts/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
```

## Alternative: Download Pre-trained Models

If you don't want to train the models yourself, you can:
1. Request the pre-trained models from the project maintainer
2. Use a cloud storage service (Google Drive, Dropbox, etc.)
3. Use Git LFS for large file storage (optional)

## Git LFS Setup (Optional)

If you want to version control these large files, install Git LFS:

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or
sudo apt-get install git-lfs  # Linux

# Initialize Git LFS
git lfs install

# Track .pkl files
git lfs track "*.pkl"

# Commit and push
git add .gitattributes
git commit -m "Track pkl files with Git LFS"
git push
```
