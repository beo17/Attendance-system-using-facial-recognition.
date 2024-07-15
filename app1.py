import streamlit as st
import pandas as pd
import time 
import os
from datetime import datetime
import numpy as np
from streamlit_autorefresh import st_autorefresh
st.set_page_config(
    page_title="My Streamlit App",
    layout="wide",
)
ts=time.time()
#date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
date = "19-05-2024"
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")

folder_path = "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Attendance"  # Thay thế đường dẫn này bằng đường dẫn thư mục của bạn
days = []
months=[]
for file_name in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file_name)):
        # Trích xuất ngày từ tên file
        day = file_name.split("_")[1].split(".")[0]  # Giả sử ngày được lưu trong tên file trước dấu chấm
        month=file_name.split("-")[1].split("-")[0]
        days.append(day)
        months.append(month)



col1, col2 = st.columns([5, 5])
with col1:
    st.title("Điểm danh học sinh")
    st.header(date)
    st.markdown("Xem học sinh trên xe: [click](http://192.168.137.136/)")
with col2:
    st.image("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Dataset\\logobachkhoatoi.png", width=300)


col3, col4 = st.columns([10, 1])
# print(months)
with col3:
    # st.header('Điểm danh học sinh')
    st.subheader("Danh sách sinh viên check in" )

    df = pd.read_csv("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Attendance\\Attendance_" + date + "check-in.csv", names=["MSSV", "NAME", "TIME-IN", "ADDRESS CHECK-IN", "TIME-OUT", "ADDRESS CHECK-OUT", "PHONE"], skiprows=1)

    df.index+=1 
    total_students = df.shape[0]

    # Hiển thị tổng số sinh viên
    #st.write(f"Tổng số sinh viên: {total_students}")
    st.markdown(f"Tổng số sinh viên đã lên xe: **{total_students}**")


    #  st.dataframe(df.style.highlight_max())
    st.dataframe(df.style.apply(lambda x: ['background: yellow' if x.name == total_students else '' for i in x], axis=0))


# Định vị tiêu đề trong cột thứ hai
# with col4:
# #     st.image("Resources/logobachkhoatoi.png", width=300)

#     st.subheader("Danh sách sinh viên check out" )

#     df=pd.read_csv("Attendance/Attendance_" + date +"check-out" + ".csv", names=["MSSV", "NAME", "TIME","ADDRESS CHECK-OUT","PHONE"],skiprows=1)
#     total_students = df.shape[0]
#     df.index+=1 
#     # Hiển thị tổng số sinh viên
#     #st.write(f"Tổng số sinh viên: {total_students}")
#     st.markdown(f"Tổng số sinh viên đã xuống xe: **{total_students}**")


#     #  st.dataframe(df.style.highlight_max())
#     st.dataframe(df.style.apply(lambda x: ['background: yellow' if x.name == total_students else '' for i in x], axis=0))







count = st_autorefresh(interval=2000, limit=10000, key="fizzbuzzcounter")



if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
else:
    st.write(f"Count: {count}")



