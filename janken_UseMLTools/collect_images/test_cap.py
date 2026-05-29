import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice
)
from picamera2 import Picamera2

def catch_frame():

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as target:
        hef = HEF("yolo26n.hef")

        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        #入出力ストリーム(推論の入出力)のパラメータ設定
        input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.UINT8)
        output_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.UINT8)

        #データセットのパラメータ設定
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        input_name = input_vstream_info.name
        model_height, model_width, _ = input_vstream_info.shape
            
        camera = Picamera2()
        camera.configure(camera.create_preview_configuration(main={"format": "XRGB8888", "size": (model_width, model_height)}))
        camera.start()

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):

                while True:
                    frame = camera.capture_array()
                    if frame is None:
                        print("フレームの取得失敗")
                        break
                
                    #capture_array() = (height, width, channel)の三次元で帰ってくるため、１次元追加する
                    input_data = {input_name: np.expand_dims(frame, axis=0)}
                    #推論の開始
                    infer_results = infer_pipeline.infer(input_data)

                    for output_name, output_tensor in infer_results.items():
                        print(f"output : [{output_name}] : shape = {output_tensor.shape}, data = {output_tensor.flatten()[:5]}")
                    
                    cv2.imshow('映像はこちら', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Check keyboard intrrupt -> 終了")
                        break

                camera.close()
                cv2.destroyAllWindows()

def main():
    catch_frame()

if __name__ == "__main__":
    main()
