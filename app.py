import os
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
PROCESSOR_NAME = "google/mobilenet_v2_1.0_224"

# Download model and compatible image processor at startup.
processor = AutoImageProcessor.from_pretrained(PROCESSOR_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()


def get_recommendation(disease_name):
    name = disease_name.lower()

    if "healthy" in name:
        return (
            "The model predicts that the plant appears healthy. "
            "Continue regular monitoring and appropriate watering."
        )
    if "early blight" in name:
        return (
            "The model predicts possible early blight. Remove badly affected "
            "leaves where appropriate, avoid overhead watering, and confirm "
            "the diagnosis with a local agricultural expert before treatment."
        )
    if "late blight" in name:
        return (
            "The model predicts possible late blight. Monitor affected plants "
            "and seek local agricultural guidance promptly before treatment."
        )
    if "bacterial" in name:
        return (
            "The model predicts a possible bacterial disease. Clean tools, "
            "avoid handling wet plants, and confirm the diagnosis before treatment."
        )
    if "rust" in name:
        return (
            "The model predicts a possible rust disease. Monitor spread and "
            "seek crop-specific guidance before choosing a treatment."
        )
    return (
        "This is an AI-generated prediction, not a confirmed diagnosis. "
        "Please confirm it with an agricultural expert before taking action."
    )


def predict_leaf(image):
    if image is None:
        return "Please upload a leaf image.", "No image uploaded."

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)[0]

    top = torch.topk(probabilities, k=3)
    rows = []
    for score, index in zip(top.values, top.indices):
        label = model.config.id2label[index.item()]
        rows.append(f"{label}: {score.item() * 100:.2f}%")

    best_score = top.values[0].item() * 100
    best_label = model.config.id2label[top.indices[0].item()]
    report = "Top 3 predictions (model scores):\n" + "\n".join(rows)

    # This is a demonstration threshold, not a clinically validated confidence cutoff.
    if best_score < 60:
        recommendation = (
            "Low-score warning: try a clear photo of one leaf and confirm the "
            "result with an agricultural expert. Do not use this prediction "
            "alone to decide on treatment."
        )
    else:
        recommendation = get_recommendation(best_label)

    return report, recommendation


with gr.Blocks(theme=gr.themes.Soft(), title="AI Crop Disease Detection") as demo:
    gr.Markdown(
        """
        # 🌱 AI-Based Crop Disease Detection
        ### Smart Farming Support System

        Upload a clear photo of a **single crop leaf** to see the model's
        top three predicted classes and basic crop-care guidance.
        """
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload crop leaf image")
        with gr.Column():
            prediction_output = gr.Textbox(label="Top 3 predictions", lines=5)
            recommendation_output = gr.Textbox(
                label="Crop-care guidance", lines=5
            )

    predict_button = gr.Button("Predict disease", variant="primary")
    predict_button.click(
        fn=predict_leaf,
        inputs=image_input,
        outputs=[prediction_output, recommendation_output],
    )

    gr.Markdown(
        """
        ---
        **Educational prototype:** Predictions may be incorrect, especially
        for photos taken in real-world conditions. The score is not a verified
        probability of disease. Confirm suspected diseases with an agricultural
        expert before applying treatments.
        """
    )

if __name__ == "__main__":
    # Render supplies PORT. The default is useful for local testing.
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
