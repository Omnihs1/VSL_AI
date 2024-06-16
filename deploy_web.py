import streamlit as st
import requests

st.header("Video API for Model")

video = st.file_uploader("Upload a video file", type=["mp4"])

if video is not None:
    st.video(video)
    response = requests.post("http://localhost:8610/video_async", files={"file" : video})
    result = response.json()
    if st.button("Recognition"):
        st.text(result)