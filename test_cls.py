import os
import glob
from get_wd import GetStdChar
from draw import inter

model = GetStdChar()
inter = inter()

f = open("curv.txt", "w")

for i in range(40):
    # if i != 4:
    #     continue
    cls_id = str(i)
    src_dir = os.path.join("../result", cls_id, "B")
    all_imgs = glob.glob(src_dir + "/" + "*.jpg")
    for img in all_imgs:
        imgname = os.path.basename(img).split(".")[0]
        idstring, stroke_id = imgname.split("_")
        stroke_id = eval(stroke_id)
        char_id, invalid_label, order_label = model.get_doc(idstring=idstring)
        user_wd = model.get_user(idstring)
        std_wd = model.get_std(char_id)
        curv = inter.single_stroke(string_id=idstring, user_wd=user_wd, std_wd=std_wd, stroke_idx=stroke_id)
        f.write(str(curv))
        f.write("\n")

f.close()
