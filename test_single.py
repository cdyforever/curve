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

idstring = "63806fddfa87b813690f1061"
char_id, invalid_label, order_label = model.get_doc(idstring=idstring)
user_wd = model.get_user(idstring)
std_wd = model.get_std(char_id)
inter.single(string_id=idstring, user_wd=user_wd, std_wd=std_wd)
