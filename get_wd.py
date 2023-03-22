# -*- coding: utf-8 -*-
import asyncio
from pymongo import MongoClient
import sys
import numpy as np
from bson.objectid import ObjectId
sys.path.append('E:/Project/calligraphy-evaluation-algorithm')
from calligraphy_evaluation.functions.preprocess.parser import ParseWriting
from calligraphy_evaluation.functions.preprocess.custom_type.writing import WritingData

class GetStdChar:

    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.parser = ParseWriting()
        client = MongoClient('mongodb://123.57.53.249:9909')
        db = client.tornado_api  
        collection_list = db.list_collection_names()
        self.mycol = db["UserChar_new"]
        return

    def get_std(self, char_id):
        std_wd = self.loop.run_until_complete(self.parser._get_cchar_template(char_id))
        return std_wd
    
    def get_user(self, idstring):
        myquery = {"_id": ObjectId(idstring)}
        mydoc = self.mycol.find(myquery)[0]
        ptss = eval(mydoc["stroke_arr"])
        arrays = []
        for pts in ptss:
            arr = np.array(pts)
            arrays.append(arr)
        user_wd = WritingData(arrays, optim=False)
        return user_wd

    def get_doc(self, idstring):
        myquery = {"_id": ObjectId(idstring)}
        mydoc = self.mycol.find(myquery)[0]
        char_id = mydoc["char_id"]
        invalid_label = mydoc["invalid_label"]
        order_label = mydoc["order_label"]
        return char_id, invalid_label, order_label