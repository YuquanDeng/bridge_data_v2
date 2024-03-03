import glob
import os
import pickle
import matplotlib.pyplot as plt
import json
import numpy as np
import cv2
import argparse
import math

one_group_folder = '/home/nil/manipulation/datasets/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/'
# one_group_folder = '/home/liyi/Manipulation/dataset/BridgeDataV2/scripted_6_18/scripted_raw/2022-12-08_pnp_rigid_objects/2022-12-08_15-22-17/'
# one_group_folder = '/home/liyi/Manipulation/dataset/2023-03-15_14-35-28/'
# one_group_folder = '/Manipulation/datasets/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/00/2023-03-15_13-35-31'
# one_group_folder = '/Manipulation/datasets/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/01/2023-03-15_13-48-43'
#
# f = 640 / (2 * math.tan(math.radians(78) / 2))
# K_C920 = [[f, 0, 320], [0, f, 240], [0, 0, 1]]
# C902
width=640
height=480
FoV = 78
focal_length = 640 / (2 * math.tan(math.radians(FoV) / 2))
# Simplified assumption, adjust based on your scenario
c_x, c_y = width / 2, height / 2  # Assuming principal point is at the center

# Simplified camera matrix
intrinsicMatrix = np.array([
    [focal_length, 0, c_x],
    [0, focal_length, c_y],
    [0, 0, 1],
])

def filter_images(traj_group_dir):
    all_obs_dict = [x for x in glob.glob(os.path.join(traj_group_dir, 'traj*/obs_dict.pkl'))]
    filtered_images = []
    for x in sorted(all_obs_dict):
        with open(x, 'rb') as f:
            obs_dict = pickle.load(f)
        z_traj = [s[2] for s in obs_dict['state']]
        min_z = min(z_traj) # TODO
        if min_z < 0.02:    # TODO
            min_idx = z_traj.index(min_z)
            img_path = x.replace('obs_dict.pkl', f'images0/im_{min_idx}.jpg')
            xyz = obs_dict['state'][min_idx][:3]
            filtered_images.append({'img_path': img_path, 'xyz': xyz})
    return filtered_images

def random_images():
    all_obs_dict = [x for x in glob.glob(os.path.join(one_group_folder, 'raw/traj_group0/traj*/obs_dict.pkl'))]
    filtered_images = []
    for x in sorted(all_obs_dict):
        with open(x, 'rb') as f:
            obs_dict = pickle.load(f)
        
        random_idx = np.random.randint(0, len(obs_dict['state']))    
        img_path = x.replace('obs_dict.pkl', f'images0/im_{random_idx}.jpg')
        xyz = obs_dict['state'][random_idx][:3]
        filtered_images.append({'img_path': img_path, 'xyz': xyz})
    return filtered_images
    
annotations = []

def onclick(event, img_path, xyz):
    ix, iy = event.xdata, event.ydata
    annotations.append({
        'img_path': img_path,
        'pixel_coordinates': (ix, iy),
        'xyz': xyz
    })
    plt.close()

def annotate_images(filtered_images, root_dir):
    annotation_path = os.path.join(root_dir, 'annotations3d.json')

    # NOTE: the same environment should use the same extrinsic matrix.
    if os.path.exists(annotation_path):
        return 
    
    for item in filtered_images:
        # import pdb;pdb.set_trace()
        img_path = item['img_path']
        xyz = item['xyz']
        img = plt.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f"Click to annotate: {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}")
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, img_path, xyz))
        plt.show()

    with open(os.path.join(root_dir, 'annotations3d.json'), 'w') as f:
        # Use default=json_util.default if you're using PyMongo or define a custom handler for numpy types
        json.dump(annotations, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def PnPsolve(root_dir):
    annotation = os.path.join(root_dir, 'annotations3d.json')
    with open(annotation, 'r') as f:
        annotations = json.load(f)
    # Placeholder for 3D world points and 2D image points
    imagePoints = np.array([x['pixel_coordinates'] for x in annotations])  # 3D points in world space
    objectPoints = np.array([x['xyz'] for x in annotations])   # 2D points in image space

    if len(imagePoints) == 0 or len(objectPoints) == 0:
        raise Exception(f'No waypoint found in this folder {root_dir}.')

    # Assuming a simple camera matrix where focal lengths f_x and f_y are known or assumed equal,
    # and the principal point (c_x, c_y) is at the image center.
    image_size = (width, height)  # Image dimensions

    # Solve for the pose without considering distortion (assuming it's negligible)
    _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, intrinsicMatrix, None)
    return rvec, tvec
    # # Convert rotation vector to rotation matrix
    # R, _ = cv2.Rodrigues(rvecs)

    # # Extrinsic matrix [R | t]
    # extrinsicMatrix = np.hstack((R, tvecs))
    # return extrinsicMatrix

def debug_projection(rvec, tvec):
    
    annotation = os.path.join(one_group_folder, 'annotations.json')
    with open(annotation, 'r') as f:
        annotations = json.load(f)
    # Placeholder for 3D world points and 2D image points
    for x in annotations:
        
        annotated_point = np.array([x['pixel_coordinates']])
        object_points = np.array([x['xyz']])
        computed_point, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsicMatrix, None)
        image_path = x['img_path']
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found: {image_path}")
            return

        # Draw the annotated point in red
        cv2.circle(img, (int(annotated_point[0, 0]), int(annotated_point[0, 1])), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(computed_point[0, 0, 0]), int(computed_point[0, 0, 1])), 5, (0, 255, 0), -1)

        cv2.imshow('Projected Image', img)
        cv2.waitKey(0)  # Wait for a key press to move to the next image
        cv2.destroyAllWindows()  # Close the window before next image
        # # Convert to RGB for displaying with matplotlib
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # # Display the image
        # plt.figure(figsize=(10, 6))
        # plt.imshow(img_rgb)
        # plt.title("Annotated (Red) vs Computed (Green) Points")
        # plt.axis('off')  # Hide axes
        # plt.show()
    
def visualize_extrinsic_matrix(rvec, tvec, overlay=False):
    all_obs_dict = [x for x in glob.glob(os.path.join(one_group_folder, 'raw/traj_group0/traj*/obs_dict.pkl'))]
    
    for obs_dict_path in sorted(all_obs_dict):
        print(obs_dict_path)
        with open(obs_dict_path, 'rb') as f:
            obs_dict = pickle.load(f)
        objectPoints = np.array([s[:3] for s in obs_dict['state']])
        imgs = []
        computed_points = []
        for frame_idx in range(len(objectPoints)):
            img_path = obs_dict_path.replace('obs_dict.pkl', f'images0/im_{frame_idx}.jpg')
            img = cv2.imread(img_path)
            print("load img", img_path)
            if img is not None:
                object_points = np.array([obs_dict['state'][frame_idx, :3]])
                computed_point, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsicMatrix, None)
                imgs.append(img)
                computed_points.append(computed_point)

        if overlay:
            img = np.average(imgs, axis=0).astype(np.uint8)
            for idx, _ in enumerate(computed_points[:-1]):
                cv2.line(img, (int(computed_points[idx][0, 0, 0]), int(computed_points[idx][0, 0, 1])), (int(computed_points[idx+1][0, 0, 0]), int(computed_points[idx+1][0, 0, 1])), (0, 0, 255 ), 2)
                cv2.circle(img, (int(computed_point[0, 0, 0]), int(computed_point[0, 0, 1])), 5, (0, 255, 0), -1)
            cv2.imshow('Projected Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            for idx, (img, computed_point) in enumerate(zip(imgs, computed_points)):
                # Draw the annotated point in red
                # cv2.circle(img, (int(annotated_point[0, 0]), int(annotated_point[0, 1])), 5, (0, 0, 255), -1)
                cv2.circle(img, (int(computed_point[0, 0, 0]), int(computed_point[0, 0, 1])), 5, (0, 255, 0), -1)

                cv2.imshow('Projected Image', img)
                cv2.waitKey(0)  # Wait for a key press to move to the next image
                cv2.destroyAllWindows()  # Close the window before next image
            else:
                print(f"Image not found: {img_path}")

def keypose_discovery(grasp_continuous, threshold=0.02):
    gripper_changed = np.zeros_like(grasp_continuous)

    gripper_closing = False
    for idx in range(1, len(grasp_continuous)):

        # detect first index of gripper close
        difference = grasp_continuous[idx-1] - grasp_continuous[idx]
        if not gripper_closing and  difference > 0 and abs(difference) > threshold:
            gripper_changed[idx] = 1
            gripper_closing = True
            continue

        # detect gripper open
        if gripper_closing and difference < 0 and abs(difference) > threshold: 
            gripper_changed[idx] = 1
            gripper_closing = False
            continue
            
        if gripper_closing:
            gripper_changed[idx] = 1
        else:
            gripper_changed[idx] = 0
    # import pdb; pdb.set_trace()

    return gripper_changed

def extract_trajectory(root_dir, dataset_dir, save_dir, traj_idx=-1, traj_group_idx=-1, save_video=False, save_image=False, representation="full_traj", use_default_extr=False):
    traj_group_idx = '*' if traj_group_idx == -1 else traj_group_idx
    traj_idx = '*' if traj_idx == -1 else traj_idx

    dataset_root_dir = os.path.join(root_dir, dataset_dir)

    traj_group_dirs = [x for x in glob.glob(os.path.join(dataset_root_dir, "raw", f'traj_group{traj_group_idx}'))]
    sorted_traj_group_dirs = sorted(traj_group_dirs, key=lambda x: int(os.path.basename(x)[10:]))

    for traj_group_dir in sorted_traj_group_dirs:

        if use_default_extr:
            # NOTE: extrinsics for toy kitchen environment. 
            rvec = np.array([[-3.16437259], [ 1.95749993], [-1.25358838]])
            tvec = np.array([[-0.13361345], [ 0.21081486] ,[ 0.04935229]])
        else:
            # TODO: write a new filtered image function that select image under directory "0 00 01 ..."
            # TODO: after annotate, store the corresponding rvec and tvec with the env name in a dictionary
            # and save as a pickle file.
            filtered_images = filter_images(traj_group_dir=traj_group_dir)
            annotate_images(filtered_images=filtered_images, root_dir=dataset_root_dir)
            rvec, tvec = PnPsolve(root_dir=dataset_root_dir)

        traj_dirs = [x for x in glob.glob(os.path.join(traj_group_dir, f'traj{traj_idx}'))]
        sorted_traj_dirs = sorted(traj_dirs, key=lambda x: int(os.path.basename(x)[4:]))

        # iterate through each trajectory
        for curr_traj_idx, traj_dir in enumerate(sorted_traj_dirs):
            lang_path = os.path.join(traj_dir, 'lang.txt')
            if os.path.exists(lang_path):
                with open(lang_path, 'r') as file:
                    lang_goal = file.readline().rstrip("\n")
            else:
                lang_goal = None

            obs_dict_path = os.path.join(traj_dir, 'obs_dict.pkl')
            print(obs_dict_path)
            with open(obs_dict_path, 'rb') as f:
                obs_dict = pickle.load(f)
            
            # import pdb; pdb.set_trace()
            objectPoints = np.array([s[:3] for s in obs_dict['state']]) # (N, 3)

            grasp_continuous = obs_dict['state'][:, -1] # (N, 1)
            gripper_close = keypose_discovery(grasp_continuous=grasp_continuous) # 1 means gripper changed, 0 means gripper didn't change.

            imgs = []
            computed_points = []
            for frame_idx in range(len(objectPoints)):
                img_path = obs_dict_path.replace('obs_dict.pkl', f'images0/im_{frame_idx}.jpg')
                img = cv2.imread(img_path)
                print("load img", img_path)

                if img is not None:
                    object_points = np.array([obs_dict['state'][frame_idx, :3]])
                    computed_point, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsicMatrix, None)
                    imgs.append(img)
                    computed_points.append(computed_point)

                    # draw the trajectory up to the current timestamp.
                    if save_video:
                        # draw text
                        if lang_goal != None:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            org = (5, 30)
                            fontScale = 1
                            color = (153, 255, 255) # yellow color
                            thickness = 2

                            img = cv2.putText(img, lang_goal, org, font,  
                            fontScale, color, thickness, cv2.LINE_AA)  
                        
                        # draw trajectory
                        color_indices = np.linspace(0, 1, len(computed_points))
                        color_indices = color_indices[::-1]
                        for idx, _ in enumerate(computed_points[:-1]):
                            if representation == "keypose_traj":
                                if not gripper_close[idx]:
                                    continue
                    
                            # full_traj or (keypose_traj and gripper close)
                            color = tuple((255 * np.array(plt.cm.jet(color_indices[idx]))[:3]).astype(int))
                            color = (int(color[0]), int(color[1]), int(color[2]))                                
                            
                            cv2.line(img, (int(computed_points[idx][0, 0, 0]), int(computed_points[idx][0, 0, 1])), 
                                        (int(computed_points[idx+1][0, 0, 0]), int(computed_points[idx+1][0, 0, 1])), color, 2)
                        
                            # color = tuple((255 * np.array(plt.cm.jet(color_indices[-1]))[:3]).astype(int))
                            # color = (int(color[0]), int(color[1]), int(color[2]))
                            # cv2.circle(img, (int(computed_point[0, 0, 0]), int(computed_point[0, 0, 1])), 5, color, -1)
            
            if save_video:
                assert save_dir != None
                height, width = imgs[0].shape[:2]
                frame_size = (width, height)
                frame_rate = 3
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                save_group_dir = traj_group_dir.replace(root_dir, save_dir)
                os.makedirs(save_group_dir, exist_ok=True)
                filename = f'traj{curr_traj_idx}_{representation}.mp4'
                save_filepath = os.path.join(save_group_dir, filename)

                video_writer = cv2.VideoWriter(save_filepath, fourcc, frame_rate, frame_size)
                for img in imgs:
                    video_writer.write(img)

                print(f"Saved video to {save_filepath}")
                print("-"*100)
                video_writer.release()

            if save_image:
                assert save_dir != None
                color_indices = np.linspace(0, 1, len(computed_points))
                color_indices = color_indices[::-1]
                reference_img = imgs[0]

                save_group_dir = traj_group_dir.replace(root_dir, save_dir)
                os.makedirs(save_group_dir, exist_ok=True)
                filename = f'traj{curr_traj_idx}_{representation}.png'
                save_filepath = os.path.join(save_group_dir, filename)

                # draw text
                if lang_goal != None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (5, 30)
                    fontScale = 1
                    color = (153, 255, 255) # yellow color
                    thickness = 2

                    reference_img = cv2.putText(reference_img, lang_goal, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 


                # draw trajectory
                if representation == "keypose_traj":
                    gripper_close_traj_idxs = np.where(gripper_close == 1)[0]
                    assert len(gripper_close_traj_idxs) != 0, "No gripper close trajectory found."
                    start_idx = gripper_close_traj_idxs[0]
                    end_idx = gripper_close_traj_idxs[-1] + 1
                    computed_points = computed_points[start_idx:end_idx]
                
                color_indices = np.linspace(0, 1, len(computed_points))
                color_indices = color_indices[::-1]
                for idx, _ in enumerate(computed_points[:-1]):
                    # if representation == "keypose_traj":
                    #     if not gripper_close[idx]:
                    #         continue
            
                    # full_traj or (keypose_traj and gripper close)
                    color = tuple((255 * np.array(plt.cm.jet(color_indices[idx]))[:3]).astype(int))
                    color = (int(color[0]), int(color[1]), int(color[2]))                                
                    
                    cv2.line(reference_img, (int(computed_points[idx][0, 0, 0]), int(computed_points[idx][0, 0, 1])), 
                                (int(computed_points[idx+1][0, 0, 0]), int(computed_points[idx+1][0, 0, 1])), color, 2)

                # start point
                color = tuple((255 * np.array(plt.cm.jet(color_indices[0]))[:3]).astype(int))
                color = (int(color[0]), int(color[1]), int(color[2]))
                cv2.circle(reference_img, (int(computed_points[0][0, 0, 0]), int(computed_points[0][0, 0, 1])), 5, color, -1)

                # end point
                color = tuple((255 * np.array(plt.cm.jet(color_indices[-1]))[:3]).astype(int))
                color = (int(color[0]), int(color[1]), int(color[2]))
                cv2.circle(reference_img, (int(computed_points[-1][0, 0, 0]), int(computed_points[-1][0, 0, 1])), 5, color, -1)

                cv2.imwrite(save_filepath, reference_img)

                print(f"Saved video to {save_filepath}")
                print("-"*100)

def get_meta_info_from_trajectory(traj_dir, rvec, tvec):
    # reference
    """
    lang_goal: language goal of the trajectory (string)
    keypose_2d: the definition of the keypose is all waypoints between gripper close and gripper open. 
                For example, in "pick up the green object and put it in the pan", the waypoints start from
                the moment when the gripper grasp the green obejct up to the point where the gripper release
                the green object.
                shape: (n, 2) where n is the number of keyposes.
    keypose_3d: shape (n, 3) where n is the number of keyposes.
    trajectory_2d: shape (N, 2) where N is the number of waypoints in the entire trajectory.
    trajectory_3d: shape (N, 3) where N is the number of waypoints in the entire trajectory.
    front_camera_extrinsics: Dictionary stores the 'rvec' and 'tvec'.
    state: x, y, z, roll, yaw, pitch, grasp continuous
    gripper_close: shape (N,) where 1 means gripper close and 0 means gripper open.
    trajectory_dir: (str) the directory of the trajectory
    """
    meta_info = {
        "lang_goal": None,
        "keypose_2d": [],
        "keypose_3d": [],
        "trajectory_2d": [],
        "trajectory_3d": [],
        "front_camera_extrinsics": {"rvec": rvec, "tvec": tvec},
        "state": None,
        "gripper_close": [],
        "trajectory_dir": traj_dir,
    }

    lang_path = os.path.join(traj_dir, 'lang.txt')
    if os.path.exists(lang_path):
        with open(lang_path, 'r') as file:
            lang_goal = file.readline().rstrip("\n")
    else:
        lang_goal = None

    obs_dict_path = os.path.join(traj_dir, 'obs_dict.pkl')
    print(obs_dict_path)
    with open(obs_dict_path, 'rb') as f:
        obs_dict = pickle.load(f)
    
    objectPoints = np.array([s[:3] for s in obs_dict['state']]) # (N, 3)
    grasp_continuous = obs_dict['state'][:, -1] # (N, 1)
    gripper_close = keypose_discovery(grasp_continuous=grasp_continuous) # 1 means gripper close, 0 means gripper open.

    for frame_idx in range(len(objectPoints)):
        img_path = obs_dict_path.replace('obs_dict.pkl', f'images0/im_{frame_idx}.jpg')
        img = cv2.imread(img_path)
        print("load img", img_path)

        if img is not None:
            object_points = np.array([obs_dict['state'][frame_idx, :3]])
            computed_point, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsicMatrix, None)
            
            meta_info["trajectory_2d"].append(computed_point.squeeze())
            meta_info["trajectory_3d"].append(object_points.squeeze())

            if gripper_close[frame_idx]:
                meta_info["keypose_2d"].append(computed_point.squeeze())
                meta_info["keypose_3d"].append(object_points.squeeze())               

    meta_info["lang_goal"] = lang_goal
    meta_info["state"] = obs_dict['state']
    meta_info["gripper_close"] = gripper_close
    meta_info["trajectory_dir"] = traj_dir

    meta_info["trajectory_2d"] = np.stack(meta_info["trajectory_2d"], axis=0)
    meta_info["trajectory_3d"] = np.stack(meta_info["trajectory_3d"], axis=0)
    meta_info["keypose_2d"] = np.stack(meta_info["keypose_2d"], axis=0)
    meta_info["keypose_3d"] = np.stack(meta_info["keypose_3d"], axis=0)

    return meta_info


def generate_meta_info( root_dir, dataset_dir, traj_group_idx, traj_idx, save_dir, use_default_extr, representation):
    traj_group_idx = '*' if traj_group_idx == -1 else traj_group_idx
    traj_idx = '*' if traj_idx == -1 else traj_idx

    dataset_root_dir = os.path.join(root_dir, dataset_dir)
    traj_group_dirs = [x for x in glob.glob(os.path.join(dataset_root_dir, "raw", f'traj_group{traj_group_idx}'))]
    sorted_traj_group_dirs = sorted(traj_group_dirs, key=lambda x: int(os.path.basename(x)[10:]))

    # iterate over a trajectory group (e.g. trajgroup0)
    for traj_group_dir in sorted_traj_group_dirs:
        if use_default_extr:
            # NOTE: extrinsics for toy kitchen environment. 
            rvec = np.array([[-3.16437259], [ 1.95749993], [-1.25358838]])
            tvec = np.array([[-0.13361345], [ 0.21081486] ,[ 0.04935229]])
        else:
            filtered_images = filter_images(traj_group_dir=traj_group_dir)
            annotate_images(filtered_images=filtered_images, root_dir=dataset_root_dir)
            rvec, tvec = PnPsolve(root_dir=dataset_root_dir)

        traj_dirs = [x for x in glob.glob(os.path.join(traj_group_dir, f'traj{traj_idx}'))]
        sorted_traj_dirs = sorted(traj_dirs, key=lambda x: int(os.path.basename(x)[4:]))

        # iterate through each trajectory (e.g. traj0, traj1)
        for curr_traj_idx, traj_dir in enumerate(sorted_traj_dirs):
            meta_info = get_meta_info_from_trajectory(traj_dir=traj_dir, rvec=rvec, tvec=tvec)

            # # sanity check
            # for k, v in meta_info.items():
            #     if isinstance(v, np.ndarray):
            #         print(k, v.shape)
            #     elif isinstance(v, str):
            #         print(k, v)
            #     elif isinstance(k, dict):
            #         print(k, k.keys())

            # save meta info
            save_group_dir = traj_group_dir.replace(root_dir, save_dir)
            os.makedirs(save_group_dir, exist_ok=True)
            filename = f'traj{curr_traj_idx}_meta_info.pkl'
            save_filepath = os.path.join(save_group_dir, filename)
            with open(save_filepath, 'wb') as file:
                pickle.dump(meta_info, file)
                print(f"Saved to {save_filepath}")
                print("-"*100)

def main():
    parser = argparse.ArgumentParser()
    # mandoo rvt2 raw data dir: /home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128
    # mandoo save data dir: /home/nil/manipulation/datasets/sanity_check
    
    # /home/nil/manipulation/datasets/raw/bridge_data_v2/datacol2_toykitchen6/many_skills/00/2023-03-11_15-09-05
    # /home/nil/manipulation/datasets/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28
    parser.add_argument("--root_dir", type=str, default="/home/nil/manipulation/datasets/raw")
    parser.add_argument("--dataset_dir", type=str, default="bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28")
    parser.add_argument("--save_dir", type=str, default="/home/nil/manipulation/datasets/sanity_check")
    parser.add_argument("--traj_group_idx", type=int, default=0)
    parser.add_argument("--traj_idx", type=int, default=-1)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--representation", type=str, default="full_traj") # {full_traj, keypose_traj}
    parser.add_argument("--use_default_extr", action="store_true")  # use default annotated extrinsics matrix
    parser.add_argument("--store_meta_info", action="store_true")
    args = parser.parse_args()

    save_dir = args.save_dir
    root_dir = args.root_dir
    dataset_dir = args.dataset_dir
    traj_group_idx = args.traj_group_idx
    traj_idx =  args.traj_idx
    save_video = args.save_video
    save_image = args.save_image
    representation = args.representation
    use_default_extr = args.use_default_extr
    store_meta_info = args.store_meta_info

    if store_meta_info:
        generate_meta_info(
            root_dir=root_dir,
            dataset_dir=dataset_dir,
            traj_group_idx=traj_group_idx,
            traj_idx=traj_idx,
            save_dir=save_dir,
            representation=representation,
            use_default_extr=use_default_extr
        )
    else:
        # debug_projection(rvec, tvec)
        extract_trajectory(
            root_dir=root_dir,
            dataset_dir=dataset_dir,
            traj_group_idx=traj_group_idx,
            traj_idx=traj_idx,
            save_video=save_video, 
            save_dir=save_dir,
            save_image=save_image,
            representation=representation,
            use_default_extr=use_default_extr
        )

if __name__ == "__main__":
    main()



