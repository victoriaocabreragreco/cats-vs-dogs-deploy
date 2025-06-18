import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('cat_vs_dogs_res.keras')
class_names = ['cat', 'dog']

def predict(img):
    try:
        img = img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return class_names[int(prediction[0] > 0.5)]
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs='label',
                    title='Cat vs Dog Classifier')

demo.launch()
