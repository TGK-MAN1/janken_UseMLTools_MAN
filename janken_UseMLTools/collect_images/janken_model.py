import numpy as np
import cv2
from hailo_platform import VDevice, HEF
from picamera2 import Picamera2
import time

def catch_frame():

    with VDevice() as target:
        #モデルの構成
        infer_model = target.create_infer_model("yolo26n.hef", "")

        with infer_model.configure() as configure_infer_model:
            bindings = configure_infer_model.create_bindings()

            #入力用バッファの確保(8bitで確保)
            model_shape = infer_model.input().shape
            input_buffer = np.empty(model_shape, dtype=np.uint8)
            bindings.input().set_buffer(input_buffer)

            #出力用のバッファの確保(16bitで確保)
            output_buffers = {}
            for output_stream in infer_model.outputs:
                output_name = output_stream.name
                output_buffers[output_name] = np.empty(output_stream.shape, dtype=np.uint16)
                bindings.output(output_name).set_buffer(output_buffers[output_name])

            #モデルが要求する画像サイズを取得(取得できなかったら今回のモデルサイズ[480 * 480]で固定)
            spatial_dims = [dim for dim in model_shape if dim > 3]
            if len(spatial_dims) >= 2:
                model_height, model_width = spatial_dims[0], spatial_dims[1]
            else:
                model_height, model_width = 480, 480

            #モデルのサイズで取ると画角が終わっているため、一旦MAXの画質で撮る
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={
                    "format": "RGB888",
                    "size": (3280, 2464)
                }
            )
            camera.configure(config)
            camera.start()

            camera.set_controls({"Saturation":1.8, "Contrast":1.3})

            print(f"推論ループの開始（サイズ: {model_width}*{model_height}）")
            while True:
                frame = camera.capture_array()
                if frame is None:
                    print("フレームの取得失敗")
                    break

                #高画質 -> モデルが要求するサイズにお直し
                cropped_frame = frame[:, 408:3288]
                resized_frame = cv2.resize(cropped_frame, (model_width, model_height), interpolation=cv2.INTER_AREA)

                if resized_frame.shape != input_buffer.shape:
                    if len(input_buffer.shape) == 4 and input_buffer.shape[0] == 1:
                        #入力を要求される4次元にする
                        input_buffer[0, :, :, :] = resized_frame
                    else:
                        input_buffer[:] = resized_frame
                else:
                    #入力に映像データをダイレクト転送
                    input_buffer[:] = resized_frame

                #Hailo-10Hで推論実行
                configure_infer_model.run([bindings], 1000)

                layers_map = [
                    ('yolo26n/conv61', 'yolo26n/conv64', 60),  # 小さな手用
                    ('yolo26n/conv77', 'yolo26n/conv80', 30),  # 中くらいの手用
                    ('yolo26n/conv91', 'yolo26n/conv94', 15)   # 大きな手用
                ]

                best_score = -1.0
                best_hand = "手が映っていません"
                best_box = None

                hand_labels = ["rock", "scissors", "paper"]

                for box_layer, cls_layer, grid_size in layers_map:
                    box_flat = output_buffers[box_layer].view(np.float16).astype(np.float32)
                    cls_flat = output_buffers[cls_layer].view(np.float16).astype(np.float32)

                    box_raw = box_flat[:grid_size * grid_size * 4].reshape(grid_size, grid_size, 4)
                    cls_raw = cls_flat[:grid_size * grid_size * 3].reshape(grid_size, grid_size, 3)

                    # nan（非数）対策
                    box_raw = np.nan_to_num(box_raw, nan=0.0)
                    cls_raw = np.nan_to_num(cls_raw, nan=0.0)

                    #一番確率の高いインデックスを探す
                    max_idx = np.unravel_index(np.argmax(cls_raw), cls_raw.shape)
                    y, x, class_id = max_idx
                    score = cls_raw[y, x, class_id]

                    # 3つの解像度の中で一番高い確率のものを採用
                    if score > best_score:
                        best_score = score
                        bbox = box_raw[y, x, :]  # [x_center, y_center, width, height]
                        best_hand = hand_labels[class_id]

                        #画面描画処理
                        cell_w = model_width / grid_size
                        cell_h = model_height / grid_size
                        
                        center_x = int((x + 0.5) * cell_w)
                        center_y = int((y + 0.5) * cell_h)
                        
                        w = int(bbox[2]) if bbox[2] > 1.0 else int(bbox[2] * model_width)
                        h = int(bbox[3]) if bbox[3] > 1.0 else int(bbox[3] * model_height)
                        
                        w = max(50, min(w, 250))
                        h = max(50, min(h, 250))

                        best_box = (center_x - w//2, center_y - h//2, center_x + w//2, center_y + h//2)

                print("--- Hailo-10H Janken Result ---")
                # score >= 50%(0.5)を対象
                if best_score > 0.5 and best_box is not None:
                    x1, y1, x2, y2 = best_box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(model_width, x2), min(model_height, y2)

                    print(f"judge : {best_hand} | 確信度: {best_score:.2f})")
                    
                    # 画面にバウンディングボックスとテキストを描画
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(resized_frame, f"{best_hand} {best_score:.2f}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                else:
                    print("手が映っていません")

                cv2.imshow('Hailo-10H Inference', resized_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("終了します")
                    break

            camera.close()
            cv2.destroyAllWindows()

def main():
    catch_frame()

if __name__ == "__main__":
    main() 

