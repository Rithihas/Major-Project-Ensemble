
import streamlit as st
import streamlit_extras.switch_page_button as switch_page




st.session_state.clear()
st.set_page_config(page_title='About_Us', layout='wide', initial_sidebar_state="collapsed")



st.markdown("<h1 style='margin-top:5px;padding-bottom:-200px !important;text-align: center; color: #4280FF; '>About Us</h1>", unsafe_allow_html=True)


# cl1,cl2,cl3 = st.columns(3)

# with cl2:
#     st.image('./images/cvr.png')

    

# # st.header("What is Alzheimer's Disease?")
# st.markdown("<h2 style='margin-top:0px;text-align: center; color: #4280FF;'>The Team</h1>", unsafe_allow_html=True)
col1,col2,col3 = st.columns([3,1,3])
st.divider()
col1,col2,col3 = st.columns([1,1,1])
with col2:
	st.image("./images/cvr_logo.png")

st.markdown("<p style='text-align:center; font-size:20px;'> This application has been developed by the students of CVR College of Engineering.</p>",unsafe_allow_html=True)
st.divider()
st.markdown("<h3 style='margin-top:0px;text-align: left; color: #4280FF;'>Goal:</h3>", unsafe_allow_html=True)
col1,col2 = st.columns([4,12])
with col1:
	st.image("./images/adni.png")
with col2:
	st.markdown("<p style='text-align:justify;font-size:18px;'>The primary motive for developing an application for early Alzheimer's detection using an ensemble of Convolutional Neural Networks (CNN) is to make a significant positive impact on the lives of individuals at risk of developing this devastating neurodegenerative disease. Early detection is crucial as it allows for timely intervention and improved patient outcomes. This can empower medical professionals with a powerful tool to identify early warning signs of Alzheimer's in their patients, allowing proactive treatment planning. By employing ensemble modeling, our goal is to enhance accuracy and introduce greater generality in prediction of Alzheimer's Disease compared to other conventional models.</p>",unsafe_allow_html=True)
st.divider()
st.markdown("<h3 style='margin-top:0px;text-align: left; color: #4280FF;'>Developers:</h3>", unsafe_allow_html=True)
col1,col2,col3 = st.columns([3,3,3])
with col1:
	st.markdown("##### Beereddy Rithihas Reddy")
with col2:
	st.markdown("##### Kukkadapu Naga Raju")
with col3:
	st.markdown("##### Vuppala Vaishnavi")
col1,col2,col3 = st.columns([3,3,3])
with col1:
	st.write("Roll No.: 20B81A05T0")
with col2:
	st.write("Roll No.: 20B81A05T1")
with col3:
	st.write("Roll No.: 20B81A05V4")
col1,col2,col3 = st.columns([3,3,3])
with col1:
	st.write("Phone No.: 7569864614")
with col2:
	st.write("Phone No.: 9392895146")
with col3:
	st.write("Phone No.: 8977500150")
col1,col2,col3 = st.columns([3,3,3])
with col1:
	st.write("rithihasz@gmail.com")
with col2:
	st.write("kukkadapunagaraju05@gmail.com")
with col3:
	st.write("vaishnavi.vuppla@gmail.com")

st.write("&nbsp;")

st.markdown("<p style='text-align:left; font-size:15px;'> Under the guidance of: </p>",unsafe_allow_html=True)
st.markdown("<p style='text-align:left; font-size:25px;'> Mrs. V.N.V.L.S Swathi</p>",unsafe_allow_html=True)
st.markdown("<p style='text-align:left; font-size:15px;'> Senior Assistant Professor, CSE Department, CVR College of Engineering.</p>",unsafe_allow_html=True)


st.divider()
st.subheader("Thank you for visiting our website!")
st.write("For any queries, Please email us through any of the addresses mentioned above.")