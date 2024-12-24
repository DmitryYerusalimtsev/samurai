import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]


class VideoConfig:
    def __init__(self, video_path, device, model_path, video_output_path, save_to_video):
        self.video_path = video_path
        self.device = device
        self.model_path = model_path
        self.video_output_path = video_output_path
        self.save_to_video = save_to_video


def tracked_objects():
    x, y, w, h = 487, 245, 110, 282
    predefined_prompts = {
        0: ((x, y, x + w, y + h), 0)
    }
    return predefined_prompts


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=args.device)
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = tracked_objects()

    frame_rate = 30
    if args.save_to_video:
        cap = cv2.VideoCapture(args.video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
        cap.release()
        height, width = loaded_frames[0].shape[:2]

        if len(loaded_frames) == 0:
            raise ValueError("No frames were loaded from the video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast(args.device.type, dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  color[obj_id % len(color)], 2)

                out.write(img)

        if args.save_to_video:
            out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"MPS: {torch.mps.is_available()}")

    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
    elif torch.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")

    config = VideoConfig(video_path="assets/hall_2_2_cut2.mp4",
                         device=torch_device,
                         model_path="sam2/checkpoints/sam2.1_hiera_base_plus.pt",
                         video_output_path="assets/output.mp4",
                         save_to_video=True)
    main(config)
