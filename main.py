import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=False)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                The dataset utilized in this project consists of approximately 87,000 RGB images of plant leaves, encompassing both healthy and diseased samples. These images are categorized into 38 distinct classes, 
                which include a diverse range of plant species and diseases.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                #### Classes Of Prediction Available: 38
                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple scab', 'Black rot', 'Cedar apple rust', 'Apple healthy',
                    'Blueberry healthy', 'Cherry Powdery mildew', 'Cherry healthy', 
                    'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust ', 
                    'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black rot', 
                    'Grape Esca(Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 
                    'Grape healthy', 'Orange Haunglongbing (Citrus greening)', 'Peach   Bacterial spot',
                    'Peach healthy', 'Pepper, bell Bacterial spot', 'Pepper, bell healthy', 
                    'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 
                    'Soybean healthy', 'Squash Powdery mildew','Strawberry Leaf scorch', 
                    'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 
                    'Tomato Late blight', 'Tomato Leaf Mold','Tomato Septoria leaf spot', 
                    'Tomato Spider mites Two-spotted spider mite','Tomato Target Spot', 
                    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus','Tomato healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))