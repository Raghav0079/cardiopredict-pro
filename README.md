# 🫀 Cardiovascular Disease Prediction System

> A comprehensive AI-powered cardiovascular disease prediction platform with professional medical reporting capabilities

A comprehensive machine learning project for cardiovascular disease prediction featuring:

- **Complete ML Pipeline**: Data analysis, model training, and evaluation with 4 algorithms
- **Interactive Gradio Interface**: Real-time web application for risk assessment
- **Multiple Model Comparison**: Logistic Regression, Random Forest, SVM, and Gradient Boosting
- **Professional PDF Reports**: Medical-grade assessment reports with detailed analytics
- **Patient Data Management**: Structured input forms with clinical parameter validation
- **Live Deployment**: Accessible via Hugging Face Spaces

**🌐 Live Demo**: [CardioPredict Pro on Hugging Face](https://raghav81-cardiopredict-pro.hf.space)

**📊 Original Analysis**: [Google Colab Notebook](https://colab.research.google.com/drive/1Vb00nps377p8m_u_DYzMhfagoitD6A35?usp=sharing)

## 🎯 Key Highlights

- **🏆 87% F1-Score**: Best-in-class performance for medical applications
- **🔬 4 ML Algorithms**: Comprehensive model comparison and consensus
- **📄 PDF Reports**: Professional medical-grade documentation
- **🌐 Live Deployment**: Instantly accessible via Hugging Face Spaces
- **⚡ Real-time**: Instant predictions with visual analytics
- **🏥 Clinical Ready**: Suitable for medical consultation documentation

## 🌟 Features

### 🤖 Machine Learning Pipeline

- **4 Trained Models**: Logistic Regression (87% F1), Random Forest (88% AUC), SVM, Gradient Boosting
- **Comprehensive Analysis**: EDA, feature correlation, model comparison in Jupyter notebook
- **Hyperparameter Tuning**: Optimized Logistic Regression with GridSearchCV
- **Performance Metrics**: Accuracy, ROC-AUC, F1-score, and confusion matrices

### 🌐 Interactive Web Interface

- **Live Deployment**: Accessible via Hugging Face Spaces
- **Real-time Predictions**: Instant risk assessment via Gradio web app
- **Multi-model Consensus**: Combines predictions from all 4 algorithms
- **Visual Analytics**: Interactive charts and confidence visualizations
- **Risk Stratification**: Clear Low/Moderate/High risk categories
- **Medical Context**: Educational disclaimers and healthcare recommendations
- **Public Access**: No installation required, works on any device
- **Patient Data Forms**: Structured input with clinical parameter validation
- **Professional PDF Reports**: Downloadable medical-grade assessment reports
- **Clinical Documentation**: Complete patient history and recommendation tracking

### 📊 Data Analysis

- **Exploratory Data Analysis**: Comprehensive visualization of cardiovascular indicators
- **Feature Engineering**: Analysis of age, cholesterol, blood pressure, heart rate patterns
- **Statistical Insights**: Correlation matrices and distribution analysis

### 📋 Professional Medical Reports

- **PDF Generation**: Medical-grade assessment reports with professional formatting
- **Patient Information**: Complete demographic and clinical parameter documentation
- **Risk Assessment Summary**: Detailed analysis with model consensus and confidence scores
- **Clinical Recommendations**: Evidence-based guidance for healthcare professionals
- **Medical Record Integration**: Unique record numbers and timestamp tracking
- **Visual Analytics**: Embedded charts and probability distributions
- **Compliance Ready**: Professional layout suitable for medical documentation

## ⚙️ Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 500MB for dependencies and models
- **Browser**: Modern web browser for interface access

### Key Dependencies
```python
gradio>=4.0.0          # Web interface framework
scikit-learn>=1.3.0    # Machine learning algorithms
pandas>=2.0.0          # Data processing
matplotlib>=3.7.0      # Data visualization
seaborn>=0.12.0        # Statistical plots
reportlab>=4.0.0       # PDF report generation
numpy>=1.24.0          # Numerical computing
```

## 🚀 Quick Start

### 🌐 Try Live Demo (Recommended)

**Instantly access the interface without installation:**
- **Live Demo**: [https://raghav81-cardiopredict-pro.hf.space](https://raghav81-cardiopredict-pro.hf.space)
- **Features**: Full functionality with all 4 ML models
- **No Setup Required**: Ready to use immediately

### 💻 Local Installation

#### Option 1: Quick Setup
```bash
# Clone the repository
git clone https://github.com/Raghav0079/cardio-vascular.git
cd cardio-vascular

# Install dependencies
pip install -r requirements.txt

# Run the enhanced interface
python gradio_interface.py
```

#### Option 2: Manual Setup
1. **Create local directory and files:**

```powershell
mkdir cardio_interface
cd cardio_interface
```

2. **Install dependencies:**

```bash
pip install gradio pandas numpy scikit-learn matplotlib seaborn reportlab
```

3. **Download and run the interface:**
   - Copy `gradio_interface.py` to your local directory
   - Run: `python gradio_interface.py`
   - Open: [http://localhost:7860](http://localhost:7860)

### 📊 Analyze Original Data

- **Jupyter Notebook**: Open `cardio_vascular.ipynb` for complete analysis
- **Dataset**: `Cardio_vascular.csv` (downloadable via the notebook)

## 📊 Input Parameters

The interface accepts the following health parameters:

### Personal Information

- **Age**: Patient age (1-100 years)
- **Sex**: Gender (0: Female, 1: Male)

### Heart-Related Symptoms

- **Chest Pain Type**:
  - 0: Typical Angina
  - 1: Atypical Angina
  - 2: Non-anginal Pain
  - 3: Asymptomatic
- **Exercise Induced Angina**: (0: No, 1: Yes)

### Clinical Measurements

- **Resting Blood Pressure**: mm Hg (80-250)
- **Cholesterol Level**: mg/dl (100-600)
- **Maximum Heart Rate**: Achieved during exercise (50-250)

### Lab Results

- **Fasting Blood Sugar**: > 120 mg/dl (0: No, 1: Yes)
- **Resting ECG Results**:
  - 0: Normal
  - 1: ST-T Wave Abnormality
  - 2: Left Ventricular Hypertrophy

### Additional Parameters

- **ST Depression**: Induced by exercise (0.0-10.0)
- **Slope**: Of peak exercise ST segment
  - 0: Upsloping
  - 1: Flat
  - 2: Downsloping
- **Number of Major Vessels**: Colored by fluoroscopy (0-3)
- **Thalassemia**:
  - 0: Normal
  - 1: Fixed Defect
  - 2: Reversible Defect
  - 3: Not described

## 🖥️ Using the Web Interface

### 🎛️ Input Parameters

The interface provides intuitive controls for:

**👤 Personal Information**

- Age slider (1-100 years)
- Sex selection (Male/Female)

**💓 Cardiovascular Symptoms**

- Chest pain type (4 categories)
- Exercise-induced angina (Yes/No)

**🩺 Clinical Measurements**

- Resting blood pressure (80-250 mmHg)
- Cholesterol level (100-600 mg/dl)
- Maximum heart rate (50-250 bpm)

**🔬 Laboratory Results**

- Fasting blood sugar levels
- ECG abnormalities
- Additional cardiac parameters

### 📋 Professional Report Generation

**📄 PDF Report Features**

- **Patient Demographics**: Complete name and assessment timestamp
- **Clinical Parameters**: All input values with medical reference ranges
- **Risk Assessment**: Comprehensive analysis with visual risk indicators
- **Model Predictions**: Individual algorithm results with confidence scores
- **Medical Recommendations**: Evidence-based clinical guidance
- **Documentation**: Medical record numbers and professional formatting

**📊 Report Analytics**

- **Risk Visualization**: Color-coded charts and probability distributions
- **Model Consensus**: Comparative analysis across all 4 algorithms
- **Clinical Context**: Parameter interpretation and medical significance
- **Follow-up Guidance**: Recommended actions and monitoring protocols

### 📊 Results Dashboard

**🎯 Risk Assessment**

- **✅ LOW RISK**: 0-1 models detect disease (Green)
- **⚡ MODERATE RISK**: 2 models detect disease (Orange)
- **⚠️ HIGH RISK**: 3+ models detect disease (Red)

**📈 Model Predictions**

- Individual confidence scores for each algorithm
- Comparative bar charts showing prediction consensus
- Probability percentages for heart disease likelihood

**💡 Medical Recommendations**

- Personalized advice based on risk level
- Healthcare consultation guidance
- Lifestyle recommendations

## 📚 Usage Examples

### Example 1: High Risk Patient
```
👤 Patient: John Doe, Age 65
💓 Symptoms: Typical Angina, Exercise-induced pain
🩺 Vitals: BP 160/95, Cholesterol 280 mg/dl, Max HR 120
📊 Result: ⚠️ HIGH RISK (3/4 models detected disease)
📄 Action: Immediate cardiology consultation recommended
```

### Example 2: Low Risk Patient
```
👤 Patient: Jane Smith, Age 35
💓 Symptoms: No chest pain, No exercise limitations
🩺 Vitals: BP 110/70, Cholesterol 180 mg/dl, Max HR 180
📊 Result: ✅ LOW RISK (0/4 models detected disease)
📄 Action: Continue healthy lifestyle, routine checkups
```

### Example 3: Moderate Risk Patient
```
👤 Patient: Mike Johnson, Age 50
💓 Symptoms: Atypical chest pain, Occasional discomfort
🩺 Vitals: BP 140/85, Cholesterol 220 mg/dl, Max HR 150
📊 Result: ⚡ MODERATE RISK (2/4 models detected concerns)
📄 Action: Follow-up testing and lifestyle modifications
```

## 🛠️ Technical Details

### Models Used
1. **Logistic Regression**: Optimized with C=1, liblinear solver
2. **Random Forest**: Ensemble of decision trees
3. **SVM**: Support Vector Machine with probability estimation
4. **Gradient Boosting**: Sequential ensemble learning

### 📈 Model Performance (Test Set Results)

| Model | Accuracy | ROC-AUC | F1-Score | Best For |
|-------|----------|---------|-----------|----------|
| **Logistic Regression** | 84% | 86% | **87%** | **Overall Balance** |
| **Random Forest** | 77% | **88%** | 81% | **Discrimination** |
| **SVM** | 67% | 76% | 74% | Feature Learning |
| **Gradient Boosting** | 82% | 86% | 86% | Complex Patterns |

**🏆 Best Model**: Logistic Regression (highest F1-score for medical applications)

**🎯 Key Insight**: High F1-score crucial for minimizing false negatives in heart disease detection

#### 📊 Detailed Performance Analysis

| Metric | Logistic Regression | Random Forest | SVM | Gradient Boosting |
|--------|-------------------|---------------|-----|-------------------|
| **Accuracy** | 84% | 77% | 67% | 82% |
| **Precision** | 85% | 79% | 71% | 83% |
| **Recall** | 89% | 84% | 78% | 88% |
| **F1-Score** | **87%** | 81% | 74% | 86% |
| **ROC-AUC** | 86% | **88%** | 76% | 86% |
| **Training Time** | 0.05s | 0.15s | 0.12s | 0.08s |
| **Prediction Speed** | ⚡ Fast | 🚀 Very Fast | 🐌 Slow | ⚡ Fast |

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This tool is for educational and research purposes
- **Not Medical Advice**: Results should not replace professional medical consultation
- **Consult Healthcare Professionals**: Always seek qualified medical advice for health concerns
- **Model Limitations**: Predictions are based on available data and may not account for all factors

## 📁 Project Structure

```
cardio-vascular/
├── 📊 DATA & ANALYSIS
│   ├── Cardio_vascular.csv      # Heart disease dataset
│   └── cardio_vascular.ipynb    # Complete ML analysis notebook
│
├── 🌐 WEB INTERFACE 
│   ├── gradio_interface.py      # Enhanced Gradio app with PDF reports
│   ├── app.py                   # Hugging Face deployment script
│   ├── requirements.txt         # Python dependencies
│   └── launch.py                # Setup and launch script
│
├── 🚀 DEPLOYMENT
│   ├── HUGGINGFACE_DEPLOYMENT.md # Complete deployment guide
│   ├── .gitattributes          # Git LFS configuration
│   ├── spaces_config.yaml      # Hugging Face Spaces config
│   └── model_files/            # Trained model artifacts
│
├── 📖 DOCUMENTATION
│   └── README.md                # This comprehensive guide
│
└── 🔧 CONFIGURATION
    └── .gitignore               # Git ignore rules
```

### 🌐 Live Deployment
**Hugging Face Spaces**: [https://raghav81-cardiopredict-pro.hf.space](https://raghav81-cardiopredict-pro.hf.space)

### 💾 Local Setup Files
When you run the interface locally, these files are created:
- `C:/Users/{username}/cardio_interface/gradio_interface.py`
- `C:/Users/{username}/cardio_interface/requirements.txt`

## 🔧 Customization

### Adding New Models
To add new models, modify the `load_and_prepare_models()` function in `gradio_interface.py`:

```python
# Add your new model
models['Your Model'] = YourModelClass()
models['Your Model'].fit(X_train, y_train)
```

### Modifying Interface
The Gradio interface can be customized by editing the `create_interface()` function:
- Change input components
- Modify layout
- Add new visualizations
- Update styling

## 🔬 Research & Development

### 📈 Model Insights

- **Logistic Regression**: Best for medical interpretability and balanced performance
- **Random Forest**: Highest discrimination ability (ROC-AUC = 88%)
- **Hyperparameter Tuning**: GridSearchCV optimization improved model robustness
- **Feature Importance**: Age, chest pain type, and cholesterol are key predictors

### 🛠️ Technical Implementation

- **Backend**: scikit-learn for ML, pandas for data processing
- **Frontend**: Gradio for interactive web interface
- **Deployment**: Hugging Face Spaces with automatic CI/CD
- **Hosting**: Cloud-based with global accessibility
- **Visualization**: matplotlib + seaborn for statistical charts
- **PDF Generation**: ReportLab for professional medical reports
- **Data Validation**: Real-time clinical parameter validation
- **Version Control**: Git with LFS for model artifacts

## 🤝 Contributing

**🔧 Enhancement Ideas:**

- Add feature importance visualizations
- Implement additional ML models (XGBoost, Neural Networks)
- Create mobile-responsive design
- Add patient data export functionality
- Integrate with electronic health records (EHR)
- Multi-language support for global accessibility
- Patient history tracking and longitudinal analysis
- Advanced report customization and branding options
- Clinical decision support system integration
- Telemedicine platform compatibility

**📊 Research Extensions:**

- Cross-validation analysis
- External dataset validation
- Explainable AI (SHAP values)
- Uncertainty quantification

## 🏥 Medical Disclaimer

**⚠️ IMPORTANT**: This tool is for **educational and research purposes only**

- Results should **never replace professional medical diagnosis**
- Always consult qualified healthcare professionals for medical concerns
- The models are trained on limited data and may not account for all factors
- This tool does not constitute medical advice or treatment recommendations

## 📞 Support & Documentation

- **🌐 Live Demo**: [CardioPredict Pro](https://raghav81-cardiopredict-pro.hf.space)
- **📋 Deployment Guide**: See `HUGGINGFACE_DEPLOYMENT.md` for complete setup instructions
- **📧 Issues**: Report bugs or request features via GitHub Issues
- **📖 Documentation**: See inline code comments and docstrings
- **🎓 Learning**: Study the Jupyter notebook for ML methodology
- **🌐 Gradio Docs**: [Official Gradio Documentation](https://gradio.app/docs/)
- **🤗 Hugging Face**: [Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- **📄 PDF Reports**: ReportLab documentation for custom report modifications

## 🔧 Troubleshooting

### Common Issues

**🚫 "Module not found" errors**
```bash
# Solution: Install missing dependencies
pip install --upgrade -r requirements.txt
```

**🌐 Interface won't load**
```bash
# Solution: Check port availability
netstat -an | findstr :7860  # Windows
lsof -i :7860               # macOS/Linux
```

**📄 PDF generation fails**
```bash
# Solution: Install reportlab dependencies
pip install reportlab pillow
```

**🔄 Models not loading**
- Ensure sufficient RAM (2GB minimum)
- Check Python version (3.8+ required)
- Verify scikit-learn installation

### Performance Optimization

- **🚀 Faster loading**: Use SSD storage for better I/O performance
- **💾 Memory optimization**: Close other applications during training
- **🔧 CPU utilization**: Set `n_jobs=-1` for parallel processing

### Getting Help

1. **📋 Check logs**: Review console output for error messages
2. **🔍 Search issues**: Check GitHub Issues for similar problems
3. **💬 Create issue**: Provide system info and error logs
4. **📧 Contact**: Use discussion forums for general questions

---

## 🏷️ Project Status

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Version](https://img.shields.io/badge/Version-2.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange)

### 📈 Project Metrics
- **⭐ Features**: 13+ clinical parameters analyzed
- **🎯 Accuracy**: Up to 87% F1-score performance
- **🌐 Accessibility**: Zero-installation web interface
- **📄 Documentation**: Professional medical reports
- **🚀 Deployment**: Live on Hugging Face Spaces

---

**🫀 Made with ❤️ for advancing cardiovascular health through AI**

*Contributing to better heart health outcomes through accessible machine learning tools.*

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.



