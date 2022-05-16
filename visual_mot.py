import sys
import os
import logging
import colorsys
from pathlib import Path
from subprocess import call

import tqdm
import hydra
import omegaconf
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def generate_colors():
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [
        15,
        13,
        25,
        12,
        19,
        8,
        22,
        24,
        29,
        17,
        28,
        20,
        2,
        27,
        11,
        26,
        21,
        4,
        3,
        18,
        9,
        5,
        14,
        1,
        16,
        0,
        23,
        7,
        6,
        10,
    ]
    colors = [colors[idx] for idx in perm]
    return colors


def visualize_sequences(
    tracks, img_folder, output_folder, padding_num=6, create_video=True
):
    colors = generate_colors()
    max_frames_seq = tracks[:, 0].max()
    for t in tqdm.trange(1, max_frames_seq + 1):
        filename_t = str(img_folder / f"{t:0{padding_num}d}")
        if os.path.exists(filename_t + ".png"):
            filename_t = filename_t + ".png"
        elif os.path.exists(filename_t + ".jpg"):
            filename_t = filename_t + ".jpg"
        else:
            logger.warning(
                "Image file not found for " + filename_t + ".png/.jpg, continuing..."
            )
            continue
        img = cv2.imread(filename_t)

        framedata = tracks[tracks[:, 0] == t]
        for obj in framedata:
            color = np.array(colors[obj[1] % len(colors)]) * 255
            cv2.rectangle(
                img,
                obj[2:4],
                obj[2:4] + obj[4:6],
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        cv2.imwrite(str(output_folder / f"{t:0{padding_num}d}.jpg"), img)

    if create_video:
        os.chdir(output_folder)
        call(
            [
                "ffmpeg",
                "-framerate",
                "10",
                "-y",
                "-i",
                f"%0{padding_num}d.jpg",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad='width=ceil(iw/2)*2:height=ceil(ih/2)*2'",
                "output.mp4",
            ]
        )


@hydra.main(config_path="configs", config_name="visual_mot")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    logger.info(f"Configuration Parameters:\n {omegaconf.OmegaConf.to_yaml(cfg)}")

    for key in cfg.seq_name:
        tracks_file = hydra.utils.to_absolute_path(
            cfg.tracks_pattern.format(seq_name=key)
        )
        tracks = np.loadtxt(tracks_file, delimiter=",").astype(int)
        if tracks.shape[1] > 6:
            tracks = tracks[(tracks[:, 7] <= 7)]
        img_folder = Path(
            hydra.utils.to_absolute_path(cfg.img_pattern.format(seq_name=key))
        )
        output_folder = Path(
            hydra.utils.to_absolute_path(cfg.output_pattern.format(seq_name=key))
        )
        assert img_folder.exists()
        output_folder.mkdir(parents=True, exist_ok=True)
        visualize_sequences(tracks, img_folder, output_folder, cfg.padding_num)


if __name__ == "__main__":
    main()
