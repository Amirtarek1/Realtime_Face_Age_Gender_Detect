# Realtime Face Age and Gender Detection

This project implements a real-time face detection system that predicts both the age and gender of individuals using a convolutional neural network (CNN). The model utilizes pre-trained networks like VGG16 and is built with TensorFlow/Keras for accurate predictions.

## Features

- **Real-time Face Detection**: Uses OpenCV to capture video input and detect faces in real-time.
- **Age Prediction**: Predicts the age of the detected face based on a pre-trained model.
- **Gender Prediction**: Predicts the gender (Male/Female) of the detected face.
- **Emotion Detection**: Detects facial emotions from the input image (if included in the scope of the project).
  
## Requirements

- Python 3.x
- TensorFlow >= 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Amirtarek1/Realtime_Face_Age_Gender_Detection.git
    cd Realtime_Face_Age_Gender_Detection
    ```

2. Install the required dependencies


## Usage

1. Run the main StreamlitFile (EmotionProject.py) for real-time detection:

    ```bash
    streamlit run EmotionProject.py
    ```

2. The webcam feed will open, and the model will detect faces, predicting both age and gender in real-time.

## Model Details

- **Age Prediction Model**: A CNN trained on a dataset that predicts continuous age values.
- **Gender Prediction Model**: A CNN trained to predict binary gender classification (Male/Female).

## Example

Here's an example of how the system works:

- When you open the webcam feed, the model will automatically detect faces and display the predicted age and gender on the screen.

## Contributing

If you'd like to contribute to this project, feel free to fork it and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to modify and expand this README based on your specific project details, any other features, and any additional instructions you may have for users.
