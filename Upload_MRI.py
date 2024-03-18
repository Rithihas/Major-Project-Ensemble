import tensorflow as tf
import numpy as np
import cv2
import io
import streamlit as st
import streamlit_extras.switch_page_button as switch_page
from streamlit_lottie import st_lottie
import json
from collections import Counter
from streamlit_lottie import st_lottie_spinner

def extract_edges(img):
  blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
  edges = cv2.Canny(blurred_image, 50, 150)
  byte_stream = io.BytesIO()
  success, encoded_image = cv2.imencode('.jpg', edges)
  if success:
    byte_stream.write(encoded_image.tobytes())
  byte_stream.seek(0)

  edge_img = tf.keras.utils.load_img(byte_stream,target_size=(224,224,3))
  edge_img = tf.keras.preprocessing.image.img_to_array(edge_img)
  edge_img = np.expand_dims(edge_img,axis=0)
  return edge_img

def enhance(img):
  enhanced = cv2.equalizeHist(img)
  byte_stream = io.BytesIO()
  success, encoded_image = cv2.imencode('.jpg', enhanced)
  if success:
    byte_stream.write(encoded_image.tobytes())
  byte_stream.seek(0)

  enhanced_img = tf.keras.utils.load_img(byte_stream,target_size=(224,224,3))
  enhanced_img = tf.keras.preprocessing.image.img_to_array(enhanced_img)
  enhanced_img = np.expand_dims(enhanced_img,axis=0)
  return enhanced_img

def threshold(img):
  blurred = cv2.GaussianBlur(img,(5,5),0)
  _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  byte_stream = io.BytesIO()
  success, encoded_image = cv2.imencode('.jpg', thresholded)
  if success:
    byte_stream.write(encoded_image.tobytes())
  byte_stream.seek(0)

  thresh_img = tf.keras.utils.load_img(byte_stream,target_size=(224,224,3))
  thresh_img = tf.keras.preprocessing.image.img_to_array(thresh_img)
  thresh_img = np.expand_dims(thresh_img,axis=0)
  return thresh_img

def mse(image1):
    image2 = cv2.imread('./images/NC_104.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

@st.cache_data(show_spinner=False)
def load_lotties():
    path1 = './animations/blue_brain.json'
    path2 = './animations/neural_brain.json'
    ret_val1 = None 
    ret_val2 = None 
    with open(path1,"r") as f:
        ret_val1 = json.load(f)
    with open(path2,"r") as f:
        ret_val2 = json.load(f)

    return ret_val1,ret_val2    


@st.cache_data(show_spinner=False)
def load_models():
    
    model1 = tf.keras.models.load_model("./weights/model1_new.keras",compile=False)
    model1.compile(optimizer='adam',loss='categorical_crossentropy')
  
    model2 = tf.keras.models.load_model("./weights/model2_new.keras",compile=False)
    model2.compile(optimizer='adam',loss='categorical_crossentropy')

    model3 = tf.keras.models.load_model("./weights/model3_new.keras",compile=False)
    model3.compile(optimizer='adam',loss='categorical_crossentropy')

    return model1,model2,model3
   

st.set_page_config(page_title='home', layout='wide', initial_sidebar_state="collapsed")




brain_anim,load_anim = load_lotties()
   
 
if 'loaded' not in st.session_state:
        with st_lottie_spinner(load_anim,height=500):
            model1,model2,model3 = load_models()

else:
   
   model1,model2,model3 = load_models()
   

st.session_state['loaded'] = 'True'

class_dict = {0:"Alzheimer's Disease", 1:'Mild Cognitive Impairment', 2:'Healthy'}


st.markdown("<h2 style='margin-top:5px;padding-bottom:-200px !important;text-align: center; '>Upload MRI Scan</h1>", unsafe_allow_html=True)


first,second = st.columns([0.65,0.35])

with first:
    st.write("&nbsp;")
    st.write("&nbsp;")
    st.write("&nbsp;")
    st.write("&nbsp;")
    img_up = st.file_uploader("",type=['jpg','jpeg','png'])

    if st.button('Process Image',use_container_width=True):
       if img_up is None:
          st.error("Please upload MRI Image.")
          if 'results' in st.session_state:
              del st.session_state['results']
       else:
            # extracting features:
          
          file_bytes = np.asarray(bytearray(img_up.read()), dtype=np.uint8)
          img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

          if mse(img) < 5770.0:

              img_edge = extract_edges(img)
              img_enhanced = enhance(img)
              img_thresholded = threshold(img)

              final_predictions = []

              # predicting outputs (single input):
              pred1 = model1.predict(img_edge)
              label1 = np.argmax(pred1)
              final_predictions.append(label1)


              pred2 = model2.predict(img_enhanced)
              label2 = np.argmax(pred2)
              final_predictions.append(label2)

              pred3 = model3.predict(img_thresholded)
              label3 = np.argmax(pred3)
              final_predictions.append(label3)

              max_predict = max(Counter(final_predictions), key=Counter(final_predictions).get, default=None)
              ensemble_output = ''

          
              if final_predictions.count(max_predict) != 1:
                  ensemble_output = class_dict[max_predict]
              else:
                  ensemble_output = class_dict[label2]

              st.session_state.results = ensemble_output
              st.write("please scroll down to view results.")   
          
          else:
              st.error("Please upload a clear MRI Image.")
              if 'results' in st.session_state:
                  del st.session_state['results']
              
              

with second:
    st_lottie(brain_anim,
          speed=1,
          loop=True,
          quality='high',
          height=500,
          width=500,
          key="home_animation")




if 'results' in st.session_state:
   placeholder = st.container()
   with placeholder:
      st.divider()
      st.markdown("<h2 style='margin-top:0px;text-align: center; color: #4280FF;'>RESULTS</h1>", unsafe_allow_html=True)
      col1,col2,col3 = st.columns([0.385,0.315,0.3])
      with col2:
          st.image(img_up,width=300)
      st.markdown("<h2 style='margin-top:0px;text-align: center;'>"+ensemble_output+"</h1>", unsafe_allow_html=True)
 
      st.markdown("<h2 style='margin-top:0px;text-align: left; color: #4280FF;'>Lifestyle Suggestions: </h1>", unsafe_allow_html=True)
      if ensemble_output == "Alzheimer's Disease":
          st.markdown('''                  
          - ##### Early Onset of Alzheimer's has been detected. It is advised to consult a doctor.
          - ##### Artificially Simulated Experiences using VR help stave cognitive decline.
          - ##### Smoking and Consumption of Alcohol must be strictly avoided.
          - ##### Eat foods that enhance your heart health-such as fish, nuts, whole grains, olive oil, etc. These foods can also protect your cognitive health.
          - ##### Ensure a safe environment and arrange for a caretaker.
                      ''',unsafe_allow_html=True)

      elif ensemble_output == 'Mild Cognitive Impairment':
          st.markdown('''               
          - ##### Remain Socially Active and Mentally Stimulated.
          - ##### Regular aerobic exercise helps in preventing or slowing cognitive decline.
          - ##### Maintain a balanced diet and reduce fat consumption.
          - ##### Ensure you attain an adequate amount of sleep for optimal well-being and cognitive function.                
                      ''',unsafe_allow_html=True)

      else:
          st.markdown('''                        
          - ##### To maintain healthy cognitive capabilities, exercise your body and mind. 
          - ##### Playing brain games like crossword puzzles,Sudoku etc boost cognitive power. 
          - ##### Learning a new language can greatly assist cognitive thinking.
          - ##### Stay intellectually curious and creative.            
                      ''',unsafe_allow_html=True)         