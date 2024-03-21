import glob
import os
import pickle
import matplotlib.pyplot as plt
import json
import numpy as np
import cv2
import argparse
import math
import copy
from multiprocessing import Pool

from trajectory_utils_bridge_v2 import generate_meta_info, annotate_extrinsics, extract_trajectory

ALL_ENV_NAMES = ['datacol2_toykitchen7', 'datacol2_toykitchen1', 'datacol2_toysink2', 'datacol2_toykitchen2', \
                 'datacol2_tabletop_dark_wood', 'datacol1_toykitchen6', 'deepthought_folding_table', 'deepthought_robot_desk', \
                'deepthought_toykitchen2', 'datacol2_laundry_machine', 'datacol2_toykitchen6', 'datacol1_toykitchen1', \
                'deepthought_toykitchen1', 'datacol2_toykitchen7_white_tray', 'datacol2_toykitchen5', 'minsky_folding_table_white_tray', \
                'datacol2_folding_table', 'datacol2_robot_desk', 'datacol2_folding_table_white_tray']

def worker(args):
    root_dir, env_name, skill_name, extrinsic_dir, start_group_number, end_group_number, save_dir = args
    generate_meta_info(
        root_dir=root_dir,
        env_name=env_name,
        skill_name=skill_name,
        extrinsic_dir=extrinsic_dir,
        start_group_number=start_group_number,
        end_group_number=end_group_number,
        save_dir=save_dir,
    )

def main():
    parser = argparse.ArgumentParser()
    # mandoo rvt2 raw data dir: /home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128
    # mandoo save data dir: /home/nil/manipulation/datasets/sanity_check
    
    # /home/nil/manipulation/datasets/raw/bridge_data_v2/datacol2_toykitchen6/many_skills/00/2023-03-11_15-09-05
    # /home/nil/manipulation/datasets/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28
    parser.add_argument("--root_dir", type=str, default="/home/nil/manipulation/datasets/raw/bridge_data_v2")
    parser.add_argument("--env_name", type=str, default="datacol1_toykitchen1")
    parser.add_argument("--skill_name", type=str, default="many_skills")
    parser.add_argument("--extrinsic_dir", type=str, default="/home/nil/manipulation/datasets/extrinsics")
    parser.add_argument("--start_group_number", type=str, default="-1")
    parser.add_argument("--end_group_number", type=str, default="100")

    # EXTRACT TRAJECTORY & META INFO
    parser.add_argument("--save_dir", type=str, default="/home/nil/manipulation/datasets/sanity_check")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--representation", type=str, default="full_traj") # {full_traj, keypose_traj}
    parser.add_argument("--store_meta_info", action="store_true")

    # ANNOTATION
    parser.add_argument("--annotate_group_number", type=str, default="-1")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--annotate_one_group", action="store_true")

    args = parser.parse_args()
    annotate = args.annotate
    store_meta_info = args.store_meta_info

    if store_meta_info:
        print("-"*50)
        print("store meta info")
        print(f"cpu count: {os.cpu_count()}")
        print("-"*50)

        # NOTE: parallelism code
        tasks = []
        for env_name in ALL_ENV_NAMES:
            skill_names = os.listdir(os.path.join(args.root_dir, env_name))
            for skill_name in skill_names:
                task = (args.root_dir, env_name, skill_name, args.extrinsic_dir, args.start_group_number, args.end_group_number, args.save_dir)
                tasks.append(task)

        with Pool(processes=os.cpu_count()) as pool:
            pool.map(worker, tasks)

        # # NOTE: This script generate meta file for all bridge v2 environment.
        # for env_name in ALL_ENV_NAMES:
        #     skill_names = os.listdir(os.path.join(args.root_dir, env_name))
        #     for skill_name in skill_names:
        #         generate_meta_info(
        #             root_dir=args.root_dir,
        #             env_name=env_name,
        #             skill_name=skill_name,
        #             extrinsic_dir=args.extrinsic_dir,
        #             start_group_number=args.start_group_number,
        #             end_group_number=args.end_group_number,
        #             save_dir=args.save_dir,
        #         )

    return
    # elif annotate:
    #     print("-"*50)
    #     print("annotate extrinsics")
    #     print("-"*50)
    #     annotate_extrinsics(
    #         root_dir=args.root_dir,
    #         env_name=args.env_name,
    #         skill_name=args.skill_name,
    #         annotate_group_number=args.annotate_group_number,
    #         extrinsic_dir=args.extrinsic_dir,
    #         annotate_one_group_for_all=args.annotate_one_group,
    #         start_group_number=args.start_group_number,
    #         end_group_number=args.end_group_number
    #     )
    # else:
    #     print("-"*50)
    #     print("extract trajectory")
    #     print("-"*50)

    #     # debug_projection(rvec, tvec)
    #     extract_trajectory(
    #         root_dir=args.root_dir,
    #         env_name=args.env_name,
    #         skill_name=args.skill_name,
    #         extrinsic_dir=args.extrinsic_dir,
    #         save_dir=args.save_dir,
    #         save_video=args.save_video,
    #         save_image=args.save_image,
    #         representation=args.representation,
    #         start_group_number=args.start_group_number,
    #         end_group_number = args.end_group_number
    #     )



if __name__ == "__main__":
    main()