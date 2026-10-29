# AI-Based Crop Disease Detection and Smart Farming System

A student project prototype that uses a pretrained MobileNetV2 image-classification model to predict plant-health classes from a leaf photograph. A Gradio interface displays the top three model predictions and basic crop-care guidance.

## Features

- Upload a single crop-leaf image.
- Predict among the model's 38 plant-health classes.
- Display the top three model scores.
- Show general crop-care guidance or a low-score warning.
- Simple browser-based interface built with Gradio.

## Technology stack

- Python
- PyTorch
- Hugging Face Transformers
- MobileNetV2 image-classification model
- Gradio

## Model

The app loads the classification model from:

`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`

It uses the image processor from:

`google/mobilenet_v2_1.0_224`

The model and processor are downloaded when the application starts, so model weights are not stored in this repository.

## Run locally

1. Install Python 3.10 or 3.11 in a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python app.py
   ```

4. Open the local URL shown in the terminal.

The first launch downloads model files and may take some time.

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, create a new **Web Service** and connect the repository.
3. Use the following settings:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
4. Select an available free instance if offered.
5. Wait for the build and test the generated URL.

Free-tier availability, memory limits, sleep behavior, and build limits can change. The model download and PyTorch memory requirements may cause a free instance to fail; check deployment logs if that happens.

## Testing

Use `results/test_results_template.csv` to record actual test results. Do not claim an accuracy percentage until it has been calculated from your own labelled test set.

## Limitations

- This is an educational prototype, not a replacement for agricultural diagnosis.
- Predictions can be incorrect, especially for photos taken outside the dataset's conditions.
- The displayed model score is not a calibrated probability that the diagnosis is correct.
- Crop-care messages are general guidance and are not treatment prescriptions.
- Any smart-farming sensor values generated separately in a notebook must be labelled as simulated unless real sensors are connected.

## Project report and notebook

Add your existing project report and exported Colab notebook if your teacher requires them. Review notebook outputs for secrets or private information before uploading.

## License and attribution

Check the model repository's license and usage terms before redistributing or deploying it. Cite the model and any datasets or research used in your project.
