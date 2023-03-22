import asyncio
import time
import json
import sys
sys.path.append('D:/Project/calligraphy-evaluation-algorithm')
from calligraphy_evaluation import EvaluatePageWriting
import cv2

def load_app_data_json(file_path):
    with open(file_path, 'rb') as f:
        app_data = json.load(f)
    return app_data

if __name__ == '__main__':

    loop = asyncio.get_event_loop()
    dicts = load_app_data_json("13657210116_121115462_2022-06-12-02_14_49.json")
    esnw = EvaluatePageWriting()

    start = time.time()
    res = loop.run_until_complete(esnw.run(dicts, images=True))
    for result in res:
        data = result["data"]
        if data is not None:
            strokes = data["strokes"]
            for stroke in strokes:
                if len(stroke["frags"]) > 0:
                    frag = stroke["frags"][0]
                    curv = frag["delta_curv"]
                    print(curv)
