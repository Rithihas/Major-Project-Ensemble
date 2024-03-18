
import streamlit as st
import streamlit_extras.switch_page_button as switch_page




st.session_state.clear()
st.set_page_config(page_title='About_Alzheimers', layout='wide', initial_sidebar_state="collapsed")



st.markdown("<h1 style='margin-top:5px;padding-bottom:-200px !important;text-align: center; '>Alzheimer's Disease</h1>", unsafe_allow_html=True)


cl1,cl2,cl3 = st.columns(3)

with cl2:
    st.image('./images/firstpic.png')

    

# st.header("What is Alzheimer's Disease?")
st.markdown("<h2 style='margin-top:0px;text-align: left; color: #4280FF;'>What is Alzheimer's Disease?</h1>", unsafe_allow_html=True)
st.markdown(''' <p style='font-size:200px; margin-top:0px;'>


                    
- ##### Alzheimer\'s disease is the most common type of dementia.
- ##### It is a progressive disease beginning with mild memory loss and possibly leading to loss of the ability to carry on a conversation and respond to the environment.
- ##### Alzheimer\'s disease involves parts of the brain that control thought, memory, and language.
- ##### It can seriously affect a person\'s ability to carry out daily activities.

                  
            
            ''',unsafe_allow_html=True)

st.markdown("<h2 style='margin-top:0px;text-align: left; color: #4280FF;'>What is known about Alzheimer's Disease?</h1>", unsafe_allow_html=True)
st.markdown(''' <p style='font-size:200px; margin-top:-0px;'>


- ##### Scientists do not yet fully understand what causes Alzheimer's disease. There likely is not a single cause but rather several factors that can affect each person differently.            
- ##### Age is the best known risk factor for Alzheimer's disease.
- ##### Family history—researchers believe that genetics may play a role in developing Alzheimer's disease. However, genes do not equal destiny. A healthy lifestyle may help reduce your risk of developing Alzheimer's disease. Two large, long term studies indicate that adequate physical activity, a nutritious diet, limited alcohol consumption, and not smoking may help people.
- ##### Changes in the brain can begin years before the first symptoms appear.
- ##### There is growing scientific evidence that healthy behaviors, which have been shown to prevent cancer, diabetes, and heart disease, may also reduce risk for subjective cognitive decline.

                  
            
            ''',unsafe_allow_html=True)

st.markdown("<h2 style='margin-top:0px;text-align: left; color: #4280FF;'>Warning Signs of Alzheimer's Disease:</h1>", unsafe_allow_html=True)

one, two = st.columns(2)

with one:
    st.markdown(''' ##### In addition to memory problems, someone with symptoms of Alzheimer's disease may experience one or more of the following:<br><br>

- ##### Memory loss that disrupts daily life, such as getting lost in a familiar place or repeating questions.<br><br>
                
- ##### Trouble handling money and paying bills.<br><br>  
     
- ##### Difficulty completing familiar tasks at home, at work or at leisure.<br><br>  
        
- ##### Decreased or poor judgment.<br><br>
         
- ##### Misplacing things and being unable to retrace steps to find them.<br><br>  
            
- ##### Changes in mood, personality, or behavior.''',unsafe_allow_html=True)

with two:
    st.image('./images/symptoms.jpg')


st.markdown("<h2 style='margin-top:20px;text-align: left; color: #4280FF;'>Rising Concerns About Alzheimer's Disease:</h1>", unsafe_allow_html=True)

pic1,pic2,pic3 = st.columns(3)

with pic1:
    st.image('./images/concern1.jpeg')
with pic2:
    st.image('./images/concern2.jpeg')    
with pic3:
    st.image('./images/concern3.jpeg')    


st.markdown("<h2 style='margin-top:20px;text-align: left; color: #4280FF;'>Benefits Of Early Diagnosis:</h1>", unsafe_allow_html=True)

p1,p2,p3 = st.columns(3)

with p2:
    st.image('./images/caretaker.jpg')


st.markdown(''' <p style='font-size:200px; margin-top:0px;'>


                    
- ##### Early Diagnosis of Alzheimer's Disease can allow you to seek medical help as soon as possible.
- ##### An early diagnosis makes individuals eligible for a wider variety of clinical trials, which help advance research and may provide medical benefits.
- ##### Receiving an early Alzheimer's diagnosis helps lessen anxieties about why you are experiencing symptoms. You and your family also have the opportunity to maximize your time together and access resources and support programs.
- ##### Planning ahead allows you to express your wishes about legal, financial and end-of-life decisions. You and your family will be able to review and update legal documents, discuss finances and property, and identify your care preferences.

                  
            
            ''',unsafe_allow_html=True)    