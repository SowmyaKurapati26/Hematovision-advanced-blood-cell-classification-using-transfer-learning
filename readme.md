# HematoVision: Advanced Blood Cell Classification

HematoVision is an advanced blood cell classification system that uses transfer learning to accurately classify blood cells into different categories: **basophil**, **eosinophil**, **lymphocyte**, **monocyte**, and **neutrophil**.

## 🚀 Features

- **🤖 AI-Powered Classification**: Uses pre-trained MobileNetV2 with transfer learning for accurate predictions
- **🌐 Interactive Web Interface**: Modern, responsive Flask application with drag-and-drop functionality
- **⚡ Real-time Analysis**: Instant blood cell classification with confidence scores
- **🎨 Beautiful UI/UX**: Professional, medical-grade interface with animations and visual feedback
- **📱 Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- **📊 Educational Content**: Detailed information about each blood cell type
- **🖨️ Print Reports**: Generate printable analysis reports

## 📁 Project Structure

```
projectfiles/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── readme.md             # This documentation
├── dataset-master/       # Blood cell dataset
│   ├── JPEGImages/       # Blood cell images
│   └── labels.csv        # Image labels
├── templates/
│   ├── home.html         # Interactive upload page
│   └── result.html       # Results page with animations
├── static/               # Static files and uploaded images
```

## 🛠️ Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip package manager**
- **4GB RAM minimum** (8GB recommended for training)

### Quick Setup

1. **Navigate to the project directory**
   ```bash
   cd projectfiles
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if you have the dataset)
   ```bash
   python train_model.py
   ```

4. **Start the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 🎯 Usage

### Web Application

1. **Upload Images**: 
   - Drag and drop blood cell images onto the upload area
   - Or click to browse and select files
   - Supports JPG, PNG, and other common image formats

2. **Get Results**:
   - View the predicted blood cell type
   - See confidence scores with animated progress bars
   - Read educational information about the cell type
   - Print or share your analysis results

### Training Your Own Model

1. **Download the dataset**:
   - Visit: [Kaggle Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
   - Extract to `projectfiles/dataset-master/`

2. **Run training**:
   ```bash
   python train_model.py
   ```

3. **Model saved as** `Blood Cell.h5`

## 🔧 Technical Details

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned for blood cell classification
- **Input Size**: 128x128 pixels
- **Output**: 5-class classification with confidence scores
- **Training**: Binary cross-entropy loss with Adam optimizer

### Performance Metrics
- **Training Data**: ~12,000 annotated blood cell images
- **Validation Accuracy**: Varies based on training data quality
- **Inference Time**: Real-time classification (< 1 second per image)
- **Model Size**: Optimized for web deployment

### Technology Stack
- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **UI Framework**: Custom responsive design with animations

## 🎨 User Interface Features

### Interactive Elements
- **Drag & Drop Upload**: Intuitive file upload experience
- **Real-time Feedback**: Visual progress indicators and animations
- **Responsive Design**: Optimized for all device sizes
- **Professional Styling**: Medical-grade interface design

