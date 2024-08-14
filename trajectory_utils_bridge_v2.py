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

def filter_images(traj_group_dir, all=False):
    all_obs_dict = [x for x in glob.glob(os.path.join(traj_group_dir, 'traj*/obs_dict.pkl'))]
    
    if len(all_obs_dict) < 25:
        num_imgs_per_traj = math.ceil(25 // len(all_obs_dict))
    elif len(all_obs_dict) >= 35:
        traj_idxs = np.random.randint(0, len(all_obs_dict), size=(len(all_obs_dict) // 2))
        all_obs_dict_copy = copy.deepcopy(all_obs_dict)
        all_obs_dict = []
        for traj_idx in traj_idxs:
            all_obs_dict.append(all_obs_dict_copy[traj_idx])

    filtered_images = []
    for x in sorted(all_obs_dict):
        with open(x, 'rb') as f:
            obs_dict = pickle.load(f)
        z_traj = [s[2] for s in obs_dict['state']]
        z_traj.sort()
        
        if len(all_obs_dict) >= 17:
            min_z = min(z_traj) 
            if min_z < 0.02:    
                min_idx = z_traj.index(min_z)
                img_path = x.replace('obs_dict.pkl', f'images0/im_{min_idx}.jpg')
                xyz = obs_dict['state'][min_idx][:3]
                filtered_images.append({'img_path': img_path, 'xyz': xyz})
        else:
            # NOTE: if the number of trajectory is less than 6, then annotate 4 times more images.
            z_traj.sort()
            min_x = z_traj[:num_imgs_per_traj]
            for min_num in min_x:
                if min_num < 0.02:
                    min_idx = z_traj.index(min_num)
                    img_path = x.replace('obs_dict.pkl', f'images0/im_{min_idx}.jpg')
                    xyz = obs_dict['state'][min_idx][:3]
                    filtered_images.append({'img_path': img_path, 'xyz': xyz}) 
    print("Total selected filtered images: ", len(filtered_images))
    return filtered_images

def random_images(traj_group_dir):
    all_obs_dict = [x for x in glob.glob(os.path.join(traj_group_dir, 'traj*/obs_dict.pkl'))]
    
    traj_idxs = np.arange(0, len(all_obs_dict))
    if len(all_obs_dict) >= 33:
        traj_idxs = np.random.randint(0, len(all_obs_dict), size=(len(all_obs_dict) // 2))
    
    traj_idxs = traj_idxs.tolist()

    filtered_images = []
    for idx in traj_idxs:
        x = all_obs_dict[idx]
        with open(x, 'rb') as f:
            obs_dict = pickle.load(f)

        random_idx = np.random.randint(0, len(obs_dict['state']))    
        img_path = x.replace('obs_dict.pkl', f'images0/im_{random_idx}.jpg')
        xyz = obs_dict['state'][random_idx][:3]
        filtered_images.append({'img_path': img_path, 'xyz': xyz})

    print("Total selected filtered images: ", len(filtered_images))


    return filtered_images
    
annotations = []

def onclick(event, img_path, xyz, abandon_threshold=(100, 400)):
    ix, iy = event.xdata, event.ydata
    # print("ix, iy", ix, iy)

    x_threshold, y_threshold = abandon_threshold
    if ix <= x_threshold and iy >= y_threshold:
        print(f"discard {xyz}")
        plt.close()
        return
    else:
        annotations.append({
            'img_path': img_path,
            'pixel_coordinates': (ix, iy),
            'xyz': xyz
        })
        plt.close()

def annotate_images(filtered_images, root_dir, exist_return=True, abandon_threshold=(100, 400)):
    annotation_path = os.path.join(root_dir, 'annotations3d.json')

    # NOTE: the same environment should use the same extrinsic matrix.
    if os.path.exists(annotation_path) and exist_return:
        return 
    
    for item in filtered_images:
        # import pdb;pdb.set_trace()
        img_path = item['img_path']
        xyz = item['xyz']
        img = plt.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f"Click to annotate: {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}")
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, img_path, xyz, abandon_threshold=abandon_threshold))
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

def get_extrinsics(extrinsic_dir, env_name, skill_name, group_number, date_name):
    with open(os.path.join(extrinsic_dir, f'{env_name}_extrinsics.pkl'), "rb") as file:
        extrinsics_dict = pickle.load(file)
    
    rvec = None
    tvec = None
    
    if group_number in extrinsics_dict[skill_name].keys() and date_name in extrinsics_dict[skill_name][group_number].keys():
        rvec = extrinsics_dict[skill_name][group_number][date_name]['rvec']
        tvec = extrinsics_dict[skill_name][group_number][date_name]['tvec']

    return rvec, tvec    

def annotate_for_date_dir(valid_date_dirs):
    # iterate through valid data directory and select the data directory with >= 25 trajectories if exists.
    reference_date_dir = valid_date_dirs[0]
    for date_dir in valid_date_dirs:
        traj_dirs = [x for x in glob.glob(os.path.join(date_dir, f'traj*'))]
        if len(traj_dirs) >= 25:
            reference_date_dir = date_dir
            break
    
    filtered_images = filter_images(traj_group_dir=reference_date_dir)
    if len(filtered_images) < 10:
        print(f"Only {len(filtered_images)} filtered images. choose random images.")
        filtered_images = random_images(traj_group_dir=reference_date_dir)
    
    # NOTE: overwrite if the annotate3d file already existed
    annotate_images(filtered_images=filtered_images, root_dir=reference_date_dir, exist_return=False, abandon_threshold=(100, 400))
    rvec, tvec = PnPsolve(root_dir=reference_date_dir)
    return rvec, tvec

def annotate_extrinsics(
    root_dir, 
    env_name, 
    skill_name, 
    annotate_group_number, 
    extrinsic_dir, 
    annotate_one_group_for_all=False, 
    start_group_number="-1", 
    end_group_number="100"
    ):
    all_group_numbers = os.listdir(os.path.join(root_dir, env_name, skill_name))
    # NOTE: ensure all the subdirectories can be converted to integer.
    all_group_numbers = [x for x in all_group_numbers if x.isdigit()]
    all_group_numbers.sort(key=lambda x: int(x))
    
    if annotate_one_group_for_all:
        # NOTE: annotate only one group and reused the annotate extrinsics for group within (start_group_number, end_group_number)
        date_dirs = glob.glob(os.path.join(root_dir, env_name, skill_name, annotate_group_number, "*", "raw", "traj_group0"))
        valid_date_dirs = [date_dir for date_dir in date_dirs if os.path.isdir(date_dir)]
        print('annotate group number: ', annotate_group_number)

        # Annotate
        rvec, tvec = annotate_for_date_dir(valid_date_dirs=valid_date_dirs)

        # save annotated extrinsics
        env_extr_dir = os.path.join(extrinsic_dir, f'{env_name}_extrinsics.pkl')
        if os.path.exists(env_extr_dir):
            with open(env_extr_dir, "rb") as file:
                extrinsics = pickle.load(file)
        else:
            extrinsics = {}

        if skill_name not in extrinsics.keys():
            extrinsics[skill_name] = {}

        for curr_group_number in all_group_numbers:
            if int(curr_group_number) < int(start_group_number) or int(curr_group_number) >= int(end_group_number):
                continue
            print("annotate group number: ", curr_group_number)
            extrinsics[skill_name][curr_group_number] = {}
            all_date_names = os.listdir(os.path.join(root_dir, env_name, skill_name, curr_group_number))
            for curr_date_name in all_date_names:
                extrinsics[skill_name][curr_group_number][curr_date_name] = {"rvec": rvec, "tvec": tvec}

        with open(env_extr_dir, "wb") as file:
            pickle.dump(extrinsics, file)
    else:
        for annotate_group_number in all_group_numbers:
            if int(annotate_group_number) < int(start_group_number) or int(annotate_group_number) >= int(end_group_number):
                continue
            print("annotate group number: ", annotate_group_number)
            date_dirs = glob.glob(os.path.join(root_dir, env_name, skill_name, annotate_group_number, "*", "raw", "traj_group0"))
            valid_date_dirs = [date_dir for date_dir in date_dirs if os.path.isdir(date_dir)]

            # Annotate
            rvec, tvec = annotate_for_date_dir(valid_date_dirs=valid_date_dirs)

            # save annotated extrinsics.
            env_extr_dir = os.path.join(extrinsic_dir, f'{env_name}_extrinsics.pkl')
            if os.path.exists(env_extr_dir):
                with open(env_extr_dir, "rb") as file:
                    extrinsics = pickle.load(file)
            else:
                extrinsics = {}
            if skill_name not in extrinsics.keys():
                extrinsics[skill_name] = {}

            extrinsics[skill_name][annotate_group_number] = {}
            all_date_names = os.listdir(os.path.join(root_dir, env_name, skill_name, annotate_group_number))
            for curr_date_name in all_date_names:
                extrinsics[skill_name][annotate_group_number][curr_date_name] = {"rvec": rvec, "tvec": tvec}

            with open(env_extr_dir, "wb") as file:
                pickle.dump(extrinsics, file)     

def extract_trajectory(
    root_dir,
    env_name,
    skill_name,
    extrinsic_dir=None,
    save_dir=None,
    save_video=False,
    save_image=False,
    representation="full_traj",
    start_group_number=-1,
    end_group_number=100,
    verbose=False
):
    all_group_numbers = os.listdir(os.path.join(root_dir, env_name, skill_name))
    # NOTE: ensure all the subdirectories can be converted to integer.
    all_group_numbers = [x for x in all_group_numbers if x.isdigit()]
    all_group_numbers.sort(key=lambda x: int(x))

    for group_number in all_group_numbers:
        if int(group_number) < int(start_group_number) or int(group_number) >= int(end_group_number):
            continue
        print("process group number: ", group_number)
    
        # NOTE: Example: /home/nil/manipulation/datasets/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/00/2023-03-15_13-34-28/raw/traj_group0
        date_dirs = glob.glob(os.path.join(root_dir, env_name, skill_name, group_number, "*", "raw", "traj_group0"))
        valid_date_dirs = [date_dir for date_dir in date_dirs if os.path.isdir(date_dir)]
 
        for date_dir in valid_date_dirs:
            traj_dirs = [x for x in glob.glob(os.path.join(date_dir, f'traj*'))]
            sorted_traj_dirs = sorted(traj_dirs, key=lambda x: int(os.path.basename(x)[4:]))
            date_name = date_dir.split('/')[-3]
            curr_group_number = date_dir.split('/')[-4]

            rvec, tvec = get_extrinsics(extrinsic_dir, env_name, skill_name, curr_group_number, date_name)

            if verbose:
                print("rvec", rvec, "tvec", tvec)

            # iterate through each trajectory
            for curr_traj_idx, traj_dir in enumerate(sorted_traj_dirs):
                lang_path = os.path.join(traj_dir, 'lang.txt')
                if os.path.exists(lang_path):
                    with open(lang_path, 'r') as file:
                        lang_goal = file.readline().rstrip("\n")
                else:
                    lang_goal = None

                obs_dict_path = os.path.join(traj_dir, 'obs_dict.pkl')

                if verbose:
                    print(obs_dict_path)
                with open(obs_dict_path, 'rb') as f:
                    obs_dict = pickle.load(f)
                
                objectPoints = np.array([s[:3] for s in obs_dict['state']]) # (N, 3)
                grasp_continuous = obs_dict['state'][:, -1] # (N, 1)
                gripper_close = keypose_discovery(grasp_continuous=grasp_continuous) # 1 means gripper changed, 0 means gripper didn't change.

                imgs = []
                computed_points = []
                for frame_idx in range(len(objectPoints)):
                    img_path = obs_dict_path.replace('obs_dict.pkl', f'images0/im_{frame_idx}.jpg')
                    img = cv2.imread(img_path)

                    if verbose:
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

                    save_group_dir = date_dir.replace(root_dir, save_dir)
                    os.makedirs(save_group_dir, exist_ok=True)
                    filename = f'traj{curr_traj_idx}_{representation}.mp4'
                    save_filepath = os.path.join(save_group_dir, filename)

                    video_writer = cv2.VideoWriter(save_filepath, fourcc, frame_rate, frame_size)
                    for img in imgs:
                        video_writer.write(img)

                    if verbose:
                        print(f"Saved video to {save_filepath}")
                        print("-"*100)
                    video_writer.release()

                if save_image:
                    assert save_dir != None
                    color_indices = np.linspace(0, 1, len(computed_points))
                    color_indices = color_indices[::-1]
                    reference_img = imgs[0]

                    save_group_dir = date_dir.replace(root_dir, save_dir)
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

                    if verbose:
                        print(f"Saved video to {save_filepath}")
                        print("-"*100)

def get_meta_info_from_trajectory(traj_dir, rvec, tvec, verbose=False):
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
    Return None if no lang text found.
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
        print(f"{lang_path} not found.skip.")
        return None

    obs_dict_path = os.path.join(traj_dir, 'obs_dict.pkl')

    if verbose:
        print(obs_dict_path)
    with open(obs_dict_path, 'rb') as f:
        obs_dict = pickle.load(f)
    
    objectPoints = np.array([s[:3] for s in obs_dict['state']]) # (N, 3)
    grasp_continuous = obs_dict['state'][:, -1] # (N, 1)
    gripper_close = keypose_discovery(grasp_continuous=grasp_continuous) # 1 means gripper close, 0 means gripper open.

    if not os.path.exists(obs_dict_path.replace('obs_dict.pkl', f'images0')):
        print(f"Image directory not found: {obs_dict_path.replace('obs_dict.pkl', f'images0')}. Skip.")
        return None

    for frame_idx in range(len(objectPoints)):
        # img_path = obs_dict_path.replace('obs_dict.pkl', f'images0/im_{frame_idx}.jpg')
        # img = cv2.imread(img_path)

        # if verbose:
        #     print("load img", img_path)

        # if img is not None:
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

    # set to None if not keypose trajectory found.
    if len(meta_info["keypose_2d"]) == 0 or len(meta_info["keypose_3d"]) == 0:
        print(f"no keypose trajectory found in {traj_dir}.")
        meta_info["keypose_2d"] = None
        meta_info["keypose_3d"] = None
    else:
        meta_info["keypose_2d"] = np.stack(meta_info["keypose_2d"], axis=0)
        meta_info["keypose_3d"] = np.stack(meta_info["keypose_3d"], axis=0)

    return meta_info

def generate_meta_info( 
        root_dir,
        env_name,
        skill_name,
        extrinsic_dir,
        save_dir,
        start_group_number="-1",
        end_group_number="100",
        verbose=False
    ):
    all_group_numbers = os.listdir(os.path.join(root_dir, env_name, skill_name))
    # NOTE: ensure all the subdirectories can be converted to integer.
    all_group_numbers = [x for x in all_group_numbers if x.isdigit()]
    all_group_numbers.sort(key=lambda x: int(x))

    for group_number in all_group_numbers:
        if int(group_number) < int(start_group_number) or int(group_number) >= int(end_group_number):
            continue
        print("process group number: ", group_number)

        # NOTE: Example: /home/nil/manipulation/datasets/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/00/2023-03-15_13-34-28/raw/traj_group0
        date_dirs = glob.glob(os.path.join(root_dir, env_name, skill_name, group_number, "*", "raw", "traj_group0"))
        valid_date_dirs = [date_dir for date_dir in date_dirs if os.path.isdir(date_dir)]

        for date_dir in valid_date_dirs:
            traj_dirs = [x for x in glob.glob(os.path.join(date_dir, f'traj*'))]
            sorted_traj_dirs = sorted(traj_dirs, key=lambda x: int(os.path.basename(x)[4:]))
            date_name = date_dir.split('/')[-3]
            curr_group_number = date_dir.split('/')[-4]
        
            rvec, tvec = get_extrinsics(extrinsic_dir, env_name, skill_name, curr_group_number, date_name)

            # skip if extrinsics not found.
            if rvec is None and tvec is None:
                print(f"Extrinsics not found for {date_dir}.")
                continue
            
            if verbose:
                print("rvec", rvec, "tvec", tvec)

            # iterate through each trajectory (e.g. traj0, traj1)
            for curr_traj_idx, traj_dir in enumerate(sorted_traj_dirs):
                meta_info = get_meta_info_from_trajectory(traj_dir=traj_dir, rvec=rvec, tvec=tvec)
                
                if meta_info is None:
                    print(f"Skip {traj_dir}. Meta info is None.")
                    print(f"-"*100)
                    continue

                # # sanity check
                # for k, v in meta_info.items():
                #     if isinstance(v, np.ndarray):
                #         print(k, v.shape)
                #     elif isinstance(v, str):
                #         print(k, v)
                #     elif isinstance(k, dict):
                #         print(k, k.keys())
                
                # save meta info
                save_group_dir = date_dir.replace(root_dir, save_dir)
                os.makedirs(save_group_dir, exist_ok=True)
                filename = f'traj{curr_traj_idx}_meta_info.pkl'
                save_filepath = os.path.join(save_group_dir, filename)
                with open(save_filepath, 'wb') as file:
                    pickle.dump(meta_info, file)
                
                if verbose:
                    print(f"Saved to {save_filepath}")
                    print("-"*100)

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
        print("-"*50)
        generate_meta_info(
            root_dir=args.root_dir,
            env_name=args.env_name,
            skill_name=args.skill_name,
            extrinsic_dir=args.extrinsic_dir,
            start_group_number=args.start_group_number,
            end_group_number=args.end_group_number,
            save_dir=args.save_dir,
        )
    elif annotate:
        print("-"*50)
        print("annotate extrinsics")
        print("-"*50)
        annotate_extrinsics(
            root_dir=args.root_dir,
            env_name=args.env_name,
            skill_name=args.skill_name,
            annotate_group_number=args.annotate_group_number,
            extrinsic_dir=args.extrinsic_dir,
            annotate_one_group_for_all=args.annotate_one_group,
            start_group_number=args.start_group_number,
            end_group_number=args.end_group_number
        )
    else:
        print("-"*50)
        print("extract trajectory")
        print("-"*50)

        # debug_projection(rvec, tvec)
        extract_trajectory(
            root_dir=args.root_dir,
            env_name=args.env_name,
            skill_name=args.skill_name,
            extrinsic_dir=args.extrinsic_dir,
            save_dir=args.save_dir,
            save_video=args.save_video,
            save_image=args.save_image,
            representation=args.representation,
            start_group_number=args.start_group_number,
            end_group_number = args.end_group_number
        )

if __name__ == "__main__":
    main()


