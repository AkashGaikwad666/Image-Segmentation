import cv2
import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms as T
import pytesseract
from PIL import Image
from transformers import pipeline
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 1. Load a pre-trained Mask R-CNN model for segmentation
def load_segmentation_model():
    return tf.keras.models.load_model('mask_rcnn_coco.h5')

# 2. Segment all objects within the image
def segment_objects(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks, boxes, classes = model.predict(image_rgb)
    return masks, boxes, classes

# 3. Extract each segmented object and store with unique IDs
def extract_objects(image, masks, output_dir='extracted_objects'):
    os.makedirs(output_dir, exist_ok=True)
    object_images = []
    for i, mask in enumerate(masks):
        object_image = image * np.expand_dims(mask, axis=-1)
        object_pil = Image.fromarray(object_image)
        object_pil.save(os.path.join(output_dir, f'object_{i}.png'))
        object_images.append(object_pil)
    return object_images

# 4. Identify each object using a pre-trained model
def identify_objects(object_images):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = T.Compose([T.ToTensor()])
    class_labels = []
    for img in object_images:
        img_tensor = transform(img)
        with torch.no_grad():
            prediction = model([img_tensor])[0]
        class_label = prediction['labels'][0].item()
        class_labels.append(class_label)
    return class_labels

# 5. Extract text or data from each object image
def extract_text_from_objects(object_images):
    extracted_texts = []
    for img in object_images:
        extracted_text = pytesseract.image_to_string(img)
        extracted_texts.append(extracted_text)
    return extracted_texts

# 6. Summarize the attributes of each object
def summarize_object_attributes(extracted_texts):
    summarizer = pipeline("summarization")
    summaries = []
    for text in extracted_texts:
        summary = summarizer(text)[0]["summary_text"]
        summaries.append(summary)
    return summaries

# 7. Map all extracted data and attributes to each object
def map_data(object_images, class_labels, extracted_texts, summaries):
    data_mapping = {}
    for i in range(len(object_images)):
        data_mapping[i] = {
            'image_id': f'object_{i}',
            'class': class_labels[i],
            'text': extracted_texts[i],
            'summary': summaries[i]
        }
    return data_mapping

# 8. Output the original image and a table containing all mapped data
def output_results(image, data_mapping):
    df = pd.DataFrame.from_dict(data_mapping, orient='index')
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Segmented Objects with IDs')
    plt.show()
    print(df)
    return df

# 9. Run the entire pipeline
def run_pipeline(uploaded_file):
    image = np.array(Image.open(uploaded_file))
    model = load_segmentation_model()
    masks, boxes, classes = segment_objects(image, model)
    object_images = extract_objects(image, masks)
    class_labels = identify_objects(object_images)
    extracted_texts = extract_text_from_objects(object_images)
    summaries = summarize_object_attributes(extracted_texts)
    data_mapping = map_data(object_images, class_labels, extracted_texts, summaries)
    df = output_results(image, data_mapping)
    return image, df

# 10. Streamlit UI to interact with the pipeline
def main():
    st.title("AI Image Segmentation and Object Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        segmented_image, object_data = run_pipeline(uploaded_file)
        st.image(segmented_image, caption='Segmented Image', use_column_width=True)
        st.write("Object Data Table")
        st.dataframe(object_data)

if __name__ == "__main__":
    main()
