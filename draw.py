import os
import sys
sys.path.append('D:/Project/calligraphy-evaluation-algorithm')
sys.path.append('E:/Project/calligraphy-evaluation-algorithm')
from calligraphy_evaluation.config import MODEL_DIR
from calligraphy_evaluation.core import _EvaluateWriting
from calligraphy_evaluation.functions.preprocess.custom_type.writing import WritingData, StrokeWritingData
from calligraphy_evaluation.functions.preprocess.parser import ParseSingleNormWriting, ParseSingleWriting, ParseCopybookWriting
from calligraphy_evaluation.functions.preprocess.utils import nsd_to_dmnt, dmnt_to_nsd, sdk_data_to_dms, align_cchar_wd
from calligraphy_evaluation.functions.eval.evaluator import Evaluator


import asyncio
import json
import time
import cv2
import numpy as np
from curv import GetStrokeFeature
from grade import grade_func
from bend import compute_bend, STROKE_SAMPLE_NUM

stroke_ids_dict = {}
idx = 0
for key, value in STROKE_SAMPLE_NUM.items():
    stroke_ids_dict[key] = idx
    idx += 1

def load_app_data_json(file_path):
    with open(file_path, 'rb') as f:
        app_data = json.load(f)
    return app_data

class inter:

    def __init__(self) -> None:
        self.loop = asyncio.get_event_loop()
        self.parser = ParseCopybookWriting()
        self.feat = GetStrokeFeature()
        self.eval = Evaluator()
        self.size = 1024
        return
    
    def drawv2(self, saveFlag, save_sub_dir):
        save_dir = os.path.join("../result", save_sub_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_x_dir = os.path.join(save_dir, saveFlag)
        if not os.path.exists(save_x_dir):
            os.mkdir(save_x_dir)
        empty = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
        mat = self.user_mat
        mat = 255 - mat
        mat = cv2.resize(mat, (1024, 1024), interpolation=cv2.INTER_AREA)
        empty[:, :, 1] = mat
        mat = self.std_mat
        mat = 255 - mat
        mat = cv2.resize(mat, (1024, 1024), interpolation=cv2.INTER_AREA)
        empty[:, :, 2] = mat
        for idx in range(len(self.grade)):
            grade = self.grade[idx]
            cv2.putText(empty, str(int(grade * 100)), (100, 300 + idx * 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        savefile = os.path.join(save_x_dir, self.savename)
        cv2.imwrite(savefile, empty)
        return empty
    
    # def draw(self):
    #     pts = self.pts
    #     fakes = self.fakes
    #     edges = self.edges
    #     mat = self.mat
    #     mat = 255 - mat
    #     mat = cv2.resize(mat, (1024, 1024), interpolation=cv2.INTER_AREA)
    #     empty = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    #     empty[:, :, 2] = mat
    #     for pt in pts:
    #         y, x = pt
    #         yy, xx = int(y * 1024), int(x * 1024)
    #         cv2.circle(empty, (xx, yy), 2, (0, 0, 255), 2)
    #     for pt in fakes:
    #         y, x = pt
    #         yy, xx = int(y * 1024), int(x * 1024)
    #         cv2.circle(empty, (xx, yy), 2, (255, 0, 0), 2)
    #     for pt in edges:
    #         y, x = pt
    #         yy, xx = int(y * 1024), int(x * 1024)
    #         cv2.circle(empty, (xx, yy), 2, (0, 0, 0), 2)
    #     cv2.putText(empty, str(int(self.grade1 * 100)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255))
    #     cv2.putText(empty, str(int(self.grade2 * 100)), (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255))
    #     cv2.putText(empty, str(self.delta1), (50, 550), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255))
    #     cv2.putText(empty, str(self.delta2), (50, 650), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255))
    #     savefile = os.path.join("result", self.savename)
    #     cv2.imwrite(savefile, empty)
    #     return empty
    
    def go(self, app_data, name):
        copybook_dms = sdk_data_to_dms(app_data)
        res = self.loop.run_until_complete(self.parser(copybook_dms))
        user_cchars_pos, user_cchars_wd, std_cchars_wd, cchar_id = res
        for user, std, pos in zip(user_cchars_wd, std_cchars_wd, user_cchars_pos):
            if std.label == "十":
                print(pos)
                # if pos != (0, 0, 8):
                #   continue
                stroke_seq = [0, 1]
                res = self.feat(cchar_wd=user, stro_seq=stroke_seq)
                std_res = self.feat(cchar_wd=std, stro_seq=stroke_seq)
                
                delta_curv, curv_label = compute_bend(user_curv=res[0]['curvature'], 
                                                      std_curv=std_res[0]['curvature'],
                                                      std_length=res[0]['length'])
                self.delta1 = delta_curv
                self.grade1 = grade_func(delta_curv=delta_curv)
                delta_curv, curv_label = compute_bend(user_curv=res[1]['curvature'], 
                                                      std_curv=std_res[1]['curvature'],
                                                      std_length=res[1]['length'])
                self.delta2 = delta_curv
                self.grade2 = grade_func(delta_curv=delta_curv)
                self.savename =  name + "_" + str(pos[0]) + "_" + str(pos[1]) + "_" + str(pos[2]) + ".jpg"
                self.mat = user.mat
                self.drawv2()
        return
    

    def single(self, string_id, user_wd, std_wd):
        strokes_label = std_wd.strokes_label
        stroke_seq = [i for i in range(len(strokes_label))]
        user_strokes_feat = self.feat(cchar_wd=user_wd, stro_seq=stroke_seq)
        std_strokes_feat = self.feat(cchar_wd=std_wd, stro_seq=stroke_seq)
        for idx, (user_stro_feat, std_stro_feat, stroke_label) in \
                enumerate(zip(user_strokes_feat, std_strokes_feat, strokes_label)):
            m_status, m_segs, m_result, head_status, rear_status = \
                self.eval._get_match_status(user_stro_feat, std_stro_feat, stroke_label)            
            u_segs_ratio, u_segs_radian, u_segs_curv, s_segs_ratio, s_segs_radian, s_segs_curv = m_result
            self.user_mat = user_wd.stacked_mat[idx]
            self.std_mat = std_wd.stacked_mat[idx]
            ratios = STROKE_SAMPLE_NUM[stroke_label]
            self.savename = string_id + "_{}.jpg".format(idx)
            self.grade = []
            save_sub_dir = str(stroke_ids_dict[stroke_label])
            if m_status == "mismatched":
                continue
            for i in range(len(u_segs_curv)):
                if i >= len(ratios):
                    ratio = 0
                else:
                    ratio = ratios[i]
                user_length = user_stro_feat.length
                std_length = std_stro_feat.length
                user_frag_ratio = u_segs_ratio[i]
                std_frag_ratio = s_segs_ratio[i]
                std_frag_length = std_length * std_frag_ratio
                user_frag_length = user_length * user_frag_ratio
                frag_length = max(std_frag_length, user_frag_length)
                delta_curv, curv_label = compute_bend(user_curv=u_segs_curv[i], 
                                                    std_curv=s_segs_curv[i],
                                                    frag_length=frag_length,
                                                    seg_multi=ratio)
                self.grade.append(grade_func(delta_curv=delta_curv))
            print("final grad : ", self.grade)
            if min(self.grade) < 0.6:
                saveFlag = "B"
            else:
                saveFlag = "G"
            self.drawv2(saveFlag, save_sub_dir)
        return


if __name__ == "__main__":
    import os
    import glob
    a = inter()
    files = glob.glob("src_data_06/*.json")
    for file in files:
        name = os.path.basename(file)
        name = name.split(".")[0]
        app_data = load_app_data_json(file)
        print(file)
        basename = os.path.basename(file)
        name = basename.split(".")[0]
        a.go(app_data=app_data, name=name)
