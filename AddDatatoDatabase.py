
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage
import os

# cred = credentials.Certificate("face-attendance-manage-firebase-adminsdk-lc8o9-4c94711f6b.json")
# firebase_admin.initialize_app(cred,{
#     'databaseURL':"https://face-attendance-manage-default-rtdb.firebaseio.com/",
#     'storageBucket': "face-attendance-manage.appspot.com"
# })

cred = credentials.Certificate("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\face-detect-71d15-firebase-adminsdk-du0ib-d96ef84311.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-detect-71d15-default-rtdb.firebaseio.com/",
    'storageBucket': "gs://face-detect-71d15.appspot.com"
})


#folderPath = 'Dataset\\facedata\\raw'
folderPath = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Dataset\\facedata\\raw'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []

#Lưu hình ảnh đại diện trên database 
# for path in pathList:

#     fileName = f'{folderPath}/{path}'
#     bucket = storage.bucket()
#     blob = bucket.blob(fileName)
#     blob.upload_from_filename(fileName)

ref = db.reference('Students')

data = {
    "changes":
        {
            "name": "Unknow",
            "status": "0"
        },



        "Chris Evan":
        {
            "name": "Chris Evan",
            "phone": '123456789',
            "addrSchool": "West Point Academy",
            "address": " NewYork, USA",
            "id": 1915323,
            "major": "Model and Actress",
            "starting_year": 2000,
            "total_attendance": 17,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-19 00:54:34"
        },
        "Anne Hathaway":
        {
            "name": "Anne Hathaway",
            "phone": '103456789',
            "addrSchool": "West Point Academy",
            "address": "New York, USA",
            "id": 1915324,
            "major": "Actress",
            "starting_year": 1999,
            "total_attendance": 57,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-19 00:54:34"
        },
        "Brie Larson":
        {
            "name": "Brie Larson",
            "id": 1915325,
            "addrSchool": "West Point Academy",
            "phone": '120456789',
            "address": "California, USA",
            "major": "Actress and Singer",
            "starting_year": 2004,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-19 00:54:34"
        },
        "Ronaldo":
        {
            "name": "Ronaldo",
            "id": 1915326,
            "addrSchool": "West Point Academy",
            "phone": '123056789',
            "address": "Portugal",
            "major": "Player Football",
            "starting_year": 2003,
            "total_attendance": 53,
            "standing": "G",
            "year": 20,
            "last_attendance_time": "2023-05-19 00:54:34"
        },
        "Messi":
        {
            "name": "Lionel Messi",
            "id": 1915327,
            "addrSchool": "West Point Academy",
            "phone": '123406789',
            "address": "Georgia, USA",
            "major": "Actress",
            "starting_year": 2005,
            "total_attendance": 7,
            "standing": "G",
            "year": 19,
            "last_attendance_time": "2023-05-19 00:54:34"
        },
        "Hugh Jackman":
        {
            "name": "Hugh Jackman",
            "id": 1915328,
            "addrSchool": "West Point Academy",
            "phone": '123450789',
            "address": "Sydney, Australia",
            "major": "Actor",
            "starting_year": 2003,
            "total_attendance": 79,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-19 00:54:34"
        },
        "Justin Bieber":
        {
            "name": "Justin Bieber",
            "id": 1915329,
            "addrSchool": "West Point Academy",
            "phone": '123456089',
            "address": "New York, USA",
            "major": "Singer",
            "starting_year": 2007,
            "total_attendance": 62,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-19 00:54:34"
        },
        # "Sophie Turner":
        # {
        #     "name": "Sophie Turner",
        #     "id": 1910008,
        #     "addrSchool": "West Point Academy",
        #     "phone": '123456709',
        #     "address": "Northampton, England",
        #     "major": "Actress",
        #     "starting_year": 2011,
        #     "total_attendance": 7,
        #     "standing": "G",
        #     "year": 4,
        #     "last_attendance_time": "2023-05-19 00:54:34"
        # },
        
        "Emma Watson":
        {
            "name": "Emma Watson",
            "id": 1915330,
            "addrSchool": "West Point Academy",
            "phone": '123456780',
            "address": "Pari, France",
            "major": "Actress",
            "starting_year": 2001,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-19 00:54:34"
        },
    "Leonardo Dicaprio":
        {
            "name": "Leonardo Dicaprio",
            "id": 1915331,
            "addrSchool": "West Point Academy",
            "phone": '10046789',
            "address": "Georgia, USA",
            "major": "Actor",
            "starting_year": 1998,
            "total_attendance": 21,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-18 00:54:34"
        },
   
    "Thinh":
        {
            "name": "Nguyen Tan Thinh",
            "id": 1915322,
            "addrSchool": "West Point Academy",
            "phone": '0826061494',
            "address": "Ho Chi Minh, Viet Nam",
            "major": "electronic and telemunication",
            "starting_year": 2019,
            "total_attendance": 56,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-16 00:54:34"
        },
    # "Huynh Buu Khanh":
    #     {
    #         "name": "Huynh Buu Khanh",
    #         "id": 1913719,
    #         "addrSchool": "West Point Academy",
    #         "phone": '0986463244',
    #         "address": "Ho Chi Minh, Viet Nam",
    #         "major": "electronic and telemunication",
    #         "starting_year": 2019,
    #         "total_attendance": 56,
    #         "standing": "G",
    #         "year": 4,
    #         "last_attendance_time": "2023-05-16 00:54:34"
    #     },

    "Nguyen Tan Thinh":
        {
            "name": "Nguyen Tan Thinh",
            "id": 1915322,
            "addrSchool": "West Point Academy",
            "phone": '0826061494',
            "address": "Ho Chi Minh, Viet Nam",
            "major": "electronic and telemunication",
            "starting_year": 2019,
            "total_attendance": 56,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2023-05-16 00:54:34"

        }
}

for key, value in data.items():
    ref.child(key).set(value)