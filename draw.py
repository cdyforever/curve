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
from bend import compute_bend, SEG_MULTI_COEFF

stroke_ids_dict = {}
idx = 0
for key, value in SEG_MULTI_COEFF.items():
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
    
    def drawv3(self, saveFlag, save_sub_dir):
        save_dir = os.path.join("../result_new", save_sub_dir)
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
        # mat = self.std_mat
        # mat = 255 - mat
        # mat = cv2.resize(mat, (1024, 1024), interpolation=cv2.INTER_AREA)
        # empty[:, :, 2] = mat
        for idx in range(len(self.grade)):
            grade = self.grade[idx]
            cv2.putText(empty, str(int(grade * 100)), (100, 300 + idx * 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        savefile = os.path.join(save_x_dir, self.savename)
        cv2.imwrite(savefile, empty)
        return empty
    
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
            ratios = SEG_MULTI_COEFF[stroke_label]
            self.savename = string_id + "_{}.jpg".format(idx)
            self.grade = []
            save_sub_dir = str(stroke_ids_dict[stroke_label])
            if m_status == "mismatched":
                continue
            print(user_stro_feat.corners)
            print(std_stro_feat.corners)
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
                delta_curv, curv_label = compute_bend(user_curv=u_segs_curv[i], 
                                                    std_curv=s_segs_curv[i],
                                                    frag_length=user_frag_length,
                                                    seg_multi=ratio)

                self.grade.append(grade_func(delta_curv=delta_curv))
            print("final grad : ", self.grade)
            if min(self.grade) < 0.6:
                saveFlag = "B"
            else:
                saveFlag = "G"
            self.drawv2(saveFlag, save_sub_dir)
        return
    
    def single_stroke(self, string_id, user_wd, std_wd, stroke_idx):
        stroke_label = std_wd.strokes_label[stroke_idx]
        print(stroke_label)
        user_strokes = user_wd.strokes
        std_strokes = std_wd.strokes
        stroke_seq = [i for i in range(len(std_wd.strokes_label))]
        user_stroke_feat = self.feat.single(user_strokes[stroke_idx], "user")
        std_stroke_feat = self.feat.single(std_strokes[stroke_idx], "std")

        m_status, m_segs, m_result, head_status, rear_status = \
            self.eval._get_match_status(user_stroke_feat, std_stroke_feat, stroke_label)            
        u_segs_ratio, u_segs_radian, u_segs_curv, s_segs_ratio, s_segs_radian, s_segs_curv = m_result
        print(m_segs)
        self.user_mat = user_wd.stacked_mat[stroke_idx]
        self.std_mat = std_wd.stacked_mat[stroke_idx]
        ratios = SEG_MULTI_COEFF[stroke_label]
        self.savename = string_id + "_{}.jpg".format(stroke_idx)
        self.grade = []
        save_sub_dir = str(stroke_ids_dict[stroke_label])
        for i in range(len(u_segs_curv)):
            if i >= len(ratios):
                ratio = 0
            else:
                ratio = ratios[i]
            user_length = user_stroke_feat.length
            std_length = std_stroke_feat.length
            user_frag_ratio = u_segs_ratio[i]
            std_frag_ratio = s_segs_ratio[i]
            std_frag_length = std_length * std_frag_ratio
            user_frag_length = user_length * user_frag_ratio
            delta_curv, curv_label = compute_bend(user_curv=u_segs_curv[i], 
                                                std_curv=s_segs_curv[i],
                                                frag_length=user_frag_length,
                                                seg_multi=ratio)
            self.delta_curv = delta_curv
            self.grade.append(grade_func(delta_curv=delta_curv))
        print("final grad : ", self.grade)
        if min(self.grade) < 0.6:
            saveFlag = "B"
        else:
            saveFlag = "G"
        self.drawv3(saveFlag, save_sub_dir)
        return self.delta_curv