import streamlit as st
import pandas as pd
import time 
import os
from datetime import datetime
import numpy as np
from streamlit_autorefresh import st_autorefresh
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import csv
from win32com.client import Dispatch
if not firebase_admin._apps:
    cred = credentials.Certificate("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\face-detect-71d15-firebase-adminsdk-du0ib-d96ef84311.json")
    firebase_admin.initialize_app(cred,{
        'databaseURL':"https://face-detect-71d15-default-rtdb.firebaseio.com/",
        'storageBucket': "gs://face-detect-71d15.appspot.com"
    })
st.set_page_config(
    page_title="My Streamlit App",
    layout="wide",
)
ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")

folder_path = "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Attendance"  # Thay thế đường dẫn này bằng đường dẫn thư mục của bạn
days = []
months=[]
for file_name in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file_name)):
        # Trích xuất ngày từ tên file
        day = file_name.split("_")[1].split(".")[0]  # Giả sử ngày được lưu trong tên file trước dấu chấm
        month=file_name.split("-")[1].split("-")[0]
        days.append(day)
        months.append(month)
studentInfo = db.reference('Students/changes').get() 
#cam=studentInfo['cam']

col1, col2 = st.columns([5, 5])
with col1:
    st.title("Điểm danh học sinh")
    st.header(date)
   # st.markdown(f"Xem học sinh trên xe: [click]({cam})")
with col2:
    st.image("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Dataset\\logobachkhoatoi.png", width=300)


col3, col4 = st.columns([10, 1])
# print(months)
with col3:
    # st.header('Điểm danh học sinh')
    st.subheader("Danh sách sinh viên check in" )

    df=pd.read_csv("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Attendance\\Attendance_" + date +"check-in" + ".csv", names=["MSSV", "NAME", "TIME-IN","ADDRESS CHECK-IN", "TIME-OUT","ADDRESS CHECK-OUT","PHONE NUMBER"] ,skiprows=1)
    df.index+=1 
    total_students = df.shape[0]

    # Hiển thị tổng số sinh viên
    #st.write(f"Tổng số sinh viên: {total_students}")
    st.markdown(f"Tổng số sinh viên đã lên xe: **{total_students}**")


    #  st.dataframe(df.style.highlight_max())
    st.dataframe(df.style.apply(lambda x: ['background: yellow' if x.name == total_students else '' for i in x], axis=0))





count = st_autorefresh(interval=2000, limit=10000, key="fizzbuzzcounter")



if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
else:
    st.write(f"Count: {count}")

