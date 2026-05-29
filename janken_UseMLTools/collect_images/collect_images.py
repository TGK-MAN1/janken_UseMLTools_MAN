#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#参照：https://zenn.dev/datajournal1/articles/37d30a8b54769a
import cv2
from PIL import Image
import os
import sys
import subprocess
import re
from datetime import datetime
import shutil
import threading
from pynput import keyboard

latest_image_name = "None"
save_flag = False
exit_flag = False

def on_press(key):
    global save_flag
    global exit_flag

    if key == keyboard.Key.space:
        save_flag = True
    elif key == keyboard.Key.esc:
        exit_flag = True

# listener = keyboard.Listener(on_press=on_press)
# listener.start()

#自分が接続しているカメラの型番をここに入れる
def get_camera_id(camera_MAC_number="046d:0825"):
    try:
        #デバイス検索結果の返却
        result = subprocess.check_output(['v4l2-ctl', '--list-devices'], text=True)
        #デバイス名とパスのペアを抽出する
        parts = result.split('\n\n')
        for part in parts:
            if camera_MAC_number in part:
                match = re.search(r'/dev/video(\d+)', part)
                if match:
                    found_id = int(match.group(1))
                    print(f"Found {camera_MAC_number}'s pid is /dev/video{found_id}")
                    return found_id
    except Exception as e:
        print(f"ERROR SEARCHING FOR CAMERA : {e}")

    print(f"Camera : {camera_MAC_number} is not found - > カメラが割り当てられてない可能性があります")
    return 0

def capture_images(cam_id, output_dir, num_images=100):
    global latest_image_name
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 5)
    if not cap.isOpened():
        print("カメラの使用不可状態 -> pid間違いやポートの割り当てを確認してください")
        return

    count = 0
    print(f"スペースキーで撮影、qキーで中断(終了枚数：{num_images})")
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("フレームの取得失敗")
            cv2.waitKey(100)
            continue
        cv2.imshow('Camera Preview', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"img_{timestamp}.jpg"
            file_path = os.path.join(output_dir, file_name)
            latest_path = os.path.join(output_dir, "latest.jpg")
            cv2.imwrite(file_path, frame)

            if os.path.isfile(file_path):
                #shutil.copy(file_path, latest_path)
                latest_image_name = file_name
                print(f"最新の画像 : {latest_image_name}")

            else:
                print(f"{file_name}への画像の保存に失敗しています")
                save_flag = False
                break

            count += 1
            print(f"Captured {count}/{num_images}")
            save_flag = False

        elif key == ord('q'):
            print("Check keyboard intrrupt -> 終了")
            exit_flag = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False))
    # t.daemon = True
    # t.start()

    try:
        cam_id = get_camera_id()
        capture_images(cam_id, "static/images", num_images=500)
    finally:
        listener.stop()
        sys.exit(0)
