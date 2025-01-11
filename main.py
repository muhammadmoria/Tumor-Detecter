import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="DeepTumorDetect",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Background and General Layout */
        body {
            background: linear-gradient(135deg, #141E30, #243B55);
            color: #E0E0E0;
            font-family: 'Roboto', sans-serif;
        }

        /* Main Title with Sticker */
        .main-title {
            font-size: 2.8em;
            font-weight: bold;
            color: #FFD700;
            text-align: center;
            margin-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
        }

        /* Section Title Styling */
        .section-title {
            font-size: 1.8em;
            color: #FFD700;
            font-weight: bold;
            margin-top: 15px;
            text-align: center;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
        }

        /* Card Styling */
        .card {
            background: #6B8E23;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: scale(1.02);
        }

        /* Feedback and Input Fields */
        .feedback-box {
            background: #2F2F3B;
            color: #E0E0E0;
            padding: 5px;
            border-radius: 8px;
            border: 1px solid #FFD700;
            margin-bottom: 10px;
        }

        /* Input Field Styling */
        .stTextInput > div, .stTextArea > div {
            background-color: #333945;
            color: #FFFFFF;
            border-radius: 5px;
            border: 1px solid #FFD700;
            padding: 5px;
        }

        /* Responsive Buttons with Stickers */
        .stButton>button {
            background-color: #FFD700;
            color: #2C3E50;
            font-size: 1.1em;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            border: none;
            position: relative;
            padding-left: 35px;
        }
        .stButton>button::before {
            content: "🔬";
            position: absolute;
            left: 10px;
            top: 3px;
            font-size: 1.2em;
        }
        .stButton>button:hover {
            background-color: #FFC300;
            color: #282C34;
        }

        /* Uniform Recommendation Boxes */
        .recommend-box {
            background: #333945;
            color: #FFD700;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin: 5px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        }
        .recommend-box img {
            border-radius: 5px;
            height: 180px;
            width: 100%;
            object-fit: contain;
        }

        /* Profile Buttons */
        .profile-buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 15px;
        }

        .profile-buttons a {
            text-decoration: none;
            font-weight: bold;
            color: white;
            background: #FFD700;
            padding: 10px 20px;
            border-radius: 8px;
            transition: background-color 0.3s;
        }

        .profile-buttons a:hover {
            background: #FFC300;
            color: #2C3E50;
        }

        @media (max-width: 768px) {
            .main-title { font-size: 2.2em; }
            .section-title { font-size: 1.6em; }
            .profile-buttons {
                flex-direction: column;
                gap: 8px;
            }

            .profile-buttons a {
                text-align: center;
                padding: 12px;
            }
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""<div class="main-title">🔬DeepTumorDetect</div>""", unsafe_allow_html=True)



try:
    MODEL_PATH = './model_by_dawood.h5'
    model = load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error_message = str(e)


class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

def predict_tumor(image_path):
    try:
        IMAGE_SIZE = 128
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        if class_labels[predicted_class_index] == 'notumor':
            return "No Tumor", confidence_score
        else:
            return f"Tumor: {class_labels[predicted_class_index]}", confidence_score
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None




tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home","🩺Tumor Types Info", "🧬Tumor Diagnosis", "💬 Feedback"])


with tab1:
    st.markdown(
        """
        <div style="background-color: #111111; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);">
            <h2 style="color: #00d9ff; text-align: center; font-size: 32px;">👤 About Me</h2>
            <p style="font-size: 18px; color: #cccccc; text-align: justify;">
                Hi! I'm <b>Muhammad Dawood</b>, a data scientist specializing in machine learning, deep learning, and NLP.
                My passion lies in building intelligent systems and web applications that enhance user experiences.
            </p>
            <p style="text-align: center; font-size: 16px; color: #999;">📧 Gmail: <b>muhammaddawoodmoria@gmail.com</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background-color: #111111; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);">
            <h2 style="color: #00d9ff; text-align: center; font-size: 32px;">🚀 Project Overview</h2>
            <p style="font-size: 18px; color: #cccccc; text-align: justify;">
                This application leverages a deep learning model to classify brain MRI images into four categories:
            </p>
            <ul style="font-size: 18px; color: #cccccc; padding-left: 20px;">
                <li><b>🧠 Pituitary Tumor</b></li>
                <li><b>⚡ Glioma Tumor</b></li>
                <li><b>✅ No Tumor</b></li>
                <li><b>🌟 Meningioma Tumor</b></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background-color: #111111; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);">
            <h2 style="color: #00d9ff; text-align: center; font-size: 32px;">⚙️ Technologies Used</h2>
            <ul style="font-size: 18px; color: #cccccc; list-style: none; padding-left: 10px;">
                <li>🐍 <b>Python</b> - Programming Language</li>
                <li>🤖 <b>Machine Learning</b> - Brain MRI Analysis</li>
                <li>📊 <b>Data Science</b> - Data Processing and Visualization</li>
                <li>🧠 <b>Deep Learning</b> - Classification Model</li>
                <li>💡 <b>Streamlit</b> - Interactive User Interface</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px;">
            <a href="https://github.com/muhammadmoria" target="_blank" style="margin: 10px; padding: 10px 20px; background-color: #00d9ff; color: #111111; text-decoration: none; border-radius: 8px; font-size: 16px; font-weight: bold;">GitHub</a>
            <a href="https://www.linkedin.com/in/muhammadmoria/" target="_blank" style="margin: 10px; padding: 10px 20px; background-color: #00d9ff; color: #111111; text-decoration: none; border-radius: 8px; font-size: 16px; font-weight: bold;">LinkedIn</a>
            <a href="https://muhammadmoria.github.io/portfolio-/" target="_blank" style="margin: 10px; padding: 10px 20px; background-color: #00d9ff; color: #111111; text-decoration: none; border-radius: 8px; font-size: 16px; font-weight: bold;">Portfolio</a>
            <a href="https://wa.me/923709152202" target="_blank" style="margin: 10px; padding: 10px 20px; background-color: #00d9ff; color: #111111; text-decoration: none; border-radius: 8px; font-size: 16px; font-weight: bold;">WhatsApp</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab2:

    st.markdown(
        """
        <div style="text-align: center; padding: 20px; background-color: #111111; border-radius: 12px; box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.4);">
            <h1 style="color: #00d9ff; font-size: 36px; margin-bottom: 5px;">Tumor Types</h1>
            <p style="font-size: 18px; color: #cccccc;">Explore the brain tumor types classified by our AI model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    tumor_info = {
        "🧠 Pituitary Tumor": """
        <ul>
        <li><b>Description</b>: Abnormal growths in the pituitary gland, affecting hormone production and causing health issues.</li>
        <li><b>Symptoms</b>: Vision problems, headaches, hormonal imbalances.</li>
        <li><b>Treatment</b>: Surgery, medications, or radiation therapy.</li>
        </ul>
        """,
        "⚡ Glioma": """
        <ul>
        <li><b>Description</b>: Tumors from glial cells in the brain/spinal cord. Can range from slow-growing to aggressive.</li>
        <li><b>Symptoms</b>: Seizures, headaches, memory loss, motor impairments.</li>
        <li><b>Treatment</b>: Surgery, radiation, and chemotherapy.</li>
        </ul>
        """,

        "🌟 Meningioma": """
        <ul>
        <li><b>Description</b>: Tumors in the meninges (brain/spinal cord protective layers). Often benign.</li>
        <li><b>Symptoms</b>: Headaches, vision issues, difficulty concentrating.</li>
        <li><b>Treatment</b>: Surgery or radiation therapy, depending on tumor size.</li>
        </ul>
        """,
                "✅ No Tumor Detected": """
        <ul>
        <li><b>Description</b>: A healthy scan with no visible tumors. Regular check-ups are still encouraged.</li>
        <li><b>Advice</b>: Maintain a healthy lifestyle and monitor for symptoms.</li>
        </ul>
        """,
    }


    for tumor, details in tumor_info.items():
        st.markdown(
            f"""
            <div style="margin: 20px 0; padding: 20px; background-color: #222222; border-left: 8px solid #00d9ff; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.5);">
                <h2 style="color: #00d9ff; font-size: 24px; margin-bottom: 10px;">{tumor}</h2>
                <div style="font-size: 16px; color: #f1f1f1; line-height: 1.6;">{details}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with tab3:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px; background-color: #111111; border-radius: 12px; box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);">
            <h1 style="color: #00d9ff; font-size: 36px; margin-bottom: 10px;">Diagnose Tumor</h1>
            <p style="font-size: 18px; color: #cccccc;">🔍 Upload a brain MRI image to classify it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_loaded:
        st.error(f"⚠️ Failed to load the model: {model_error_message}")
    else:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 30px; padding: 10px;">
                <label style="font-size: 16px; font-weight: bold; color: #cccccc; display: block; margin-bottom: 10px;">Upload an MRI Image</label>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            try:
                st.markdown(
                    """
                    <div style="text-align: center; margin: 30px 0;">
                        <h4 style="color: #cccccc; font-weight: bold; margin-bottom: 15px;">Uploaded Image:</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(
                    uploaded_file,
                    caption="Preview",
                    use_column_width=False,
                    width=350,  # Adjusted width for better visibility
                )

                UPLOAD_FOLDER = './uploads'
                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)
                image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("🔄 Analyzing the image..."):
                    result, confidence = predict_tumor(image_path)

                if result:
                    st.markdown(
                        f"""
                        <div style="text-align: center; margin-top: 40px; background-color: #1a1a1a; padding: 20px; border-radius: 12px; box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);">
                            <h2 style="color: #00ff7f; font-weight: bold; margin-bottom: 15px;">Prediction Complete!</h2>
                            <p style="font-size: 22px; color: #cccccc;">Result: <b>{result}</b></p>
                            <p style="font-size: 20px; color: #cccccc;">Confidence: <b>{confidence*100:.2f}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("⚠️ Prediction failed. Please try again.")

            except UnidentifiedImageError:
                st.error("⚠️ The uploaded file is not a valid image. Please upload a valid JPG, PNG, or JPEG file.")
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred: {str(e)}")

import datetime

with tab4:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px; background-color: #111111; border-radius: 12px; box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);">
            <h1 style="color: #00d9ff; font-size: 36px; margin-bottom: 10px;">💬 We Value Your Feedback!</h1>
            <p style="font-size: 18px; color: #cccccc;">Your thoughts help us improve the application.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # User input form
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px;">
            <label style="font-size: 16px; font-weight: bold; color: #cccccc; display: block; margin-bottom: 10px;">Your Name</label>
        </div>
        """,
        unsafe_allow_html=True,
    )
    name = st.text_input("", key="name_input", placeholder="Enter your name")

    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px;">
            <label style="font-size: 16px; font-weight: bold; color: #cccccc; display: block; margin-bottom: 10px;">Your Feedback</label>
        </div>
        """,
        unsafe_allow_html=True,
    )
    feedback = st.text_area("", key="feedback_input", placeholder="Share your thoughts")

    # Submit feedback button
    if st.button("🚀 Submit Feedback"):
        if name and feedback:
            with open("feedback.txt", "a") as f:
                f.write(f"{datetime.datetime.now().date()} - {name}: {feedback}\n")
            st.success("Thank you for your feedback! 💙")
        else:
            st.warning("⚠️ Please provide both your name and feedback.")

    # Display feedback
    try:
        with open("feedback.txt", "r") as f:
            st.markdown(
                """
                <div style="margin-top: 40px;">
                    <h3 style="color: #00d9ff; font-size: 28px;">User Feedback:</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for line in f.readlines():
                st.markdown(
                    f"""
                    <div style="background-color: #1a1a1a; padding: 10px; border-radius: 8px; margin: 10px 0;">
                        <p style="font-size: 16px; color: #cccccc;">{line}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except FileNotFoundError:
        st.info("No feedback has been provided yet.")

