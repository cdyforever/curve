from pymongo import MongoClient
from get_wd import GetStdChar
from draw import inter

model = GetStdChar()
inter = inter()
client = MongoClient('mongodb://123.57.53.249:9909')
db = client.tornado_api  
collection_list = db.list_collection_names()
mycol = db["UserChar_new"]
ids = mycol.distinct("_id")

idstring = "63abb0c16c42781f8fceddce"
char_id, invalid_label, order_label = model.get_doc(idstring=idstring)
user_wd = model.get_user(idstring)
std_wd = model.get_std(char_id)

import cv2
cv2.imwrite("user.jpg", user_wd.mat)
cv2.imwrite("std.jpg", std_wd.mat)
inter.single_stroke(string_id=idstring, user_wd=user_wd, std_wd=std_wd, stroke_idx=0)
