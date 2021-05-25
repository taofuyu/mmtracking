import os
import os.path as osp
#import cv2
import tempfile
from argparse import ArgumentParser

import mmcv

from mmtrack.apis import inference_mot, init_model


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    assert args.output or args.show
    # load images
    if osp.isdir(args.input):
        imgs = sorted(os.listdir(args.input))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True
    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit('/', 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images
    #appeared_id = []
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(args.input, img)
        result = inference_mot(model, img, frame_id=i)
        result = result['track_results']
        # car_result = result[3]
        # for car in car_result:
        #     video_path = args.input
        #     sample_save_path = os.path.join("/data/taofuyu/tao_dataset/car_reid/high/", video_path.split("/")[-1].split(".")[0])
        #     if not os.path.exists(sample_save_path):
        #         os.makedirs(sample_save_path)
            
        #     car_id = str(int(car[0]))
        #     car_box = car[1:-1]
        #     x_min, y_min, x_max, y_max = map(int, list(car_box))
        #     car_patch = img[y_min:y_max, x_min:x_max]
        #     h,w,_ = car_patch.shape
        #     if h <=0 or w<=0:
        #         continue
        #     if not car_id in appeared_id:
        #         cv2.imwrite(os.path.join(sample_save_path, "id_"+car_id+".jpg"), car_patch)
            
        #     appeared_id.append(car_id)
        #     track_save_path = os.path.join(sample_save_path, "id_"+car_id)
        #     if not os.path.exists(track_save_path):
        #         os.makedirs(track_save_path)
        #     cv2.imwrite(track_save_path+ "/"+ str(i)+"_"+car_id+".jpg", car_patch)
            
        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit('/', 1)[-1])
        else:
            out_file = None
        model.show_result(
            img,
            result,
            show=args.show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend=args.backend)
        prog_bar.update()

    if OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps)
        out_dir.cleanup()


if __name__ == '__main__':
    main()
