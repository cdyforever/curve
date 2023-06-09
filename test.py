import json
from pymongo import MongoClient
import cv2
from bson import ObjectId
import numpy as np
from get_wd import GetStdChar
from draw import inter


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
