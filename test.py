import json
from pymongo import MongoClient
import cv2
from bson import ObjectId
import numpy as np
from get_wd import GetStdChar
from draw import inter

STROKE_DICT = {'撇': 0, '点': 1, '横': 2, '捺': 3, '竖': 4, '提画': 5, '卧钩': 6, '反捺': 7, '右点': 8, 
                 '垂露竖': 9, '左点': 10, '平捺': 11, '平撇': 12, '弯钩': 13, '悬针竖': 14, '提': 15, '撇折': 16, 
                 '撇点': 17, '斜捺': 18, '斜撇': 19, '斜钩': 20, '横折': 21, '横折弯': 22, '横折弯钩': 23, '横折折': 24,
                 '横折折折': 25, '横折折折钩': 26, '横折折撇': 27, '横折提': 28, '横折钩': 29, '横撇': 30, 
                 '横撇弯钩': 31, '横斜钩': 32, '横钩': 33, '短撇': 34, '短横': 35, '短竖': 36, '竖弯': 37, 
                 '竖弯钩': 38, '竖折': 39, '竖折折': 40, '竖折折钩': 41, '竖折撇': 42, '竖提': 43, '竖撇': 44, 
                 '竖钩': 45, '长横': 46}


model = GetStdChar()
inter = inter()
client = MongoClient('mongodb://123.57.53.249:9909')
db = client.tornado_api  
collection_list = db.list_collection_names()
mycol = db["UserChar_new"]
ids = mycol.distinct("_id")
for id in ids:
    idstring = str(id)
    print(idstring)
    myquery = {"_id": id}
    char_id, invalid_label, order_label = model.get_doc(idstring=idstring)
    if invalid_label != 0:
        continue
    if order_label != True:
        continue
    user_wd = model.get_user(idstring)
    std_wd = model.get_std(char_id)
    inter.single(string_id=idstring, user_wd=user_wd, std_wd=std_wd)
