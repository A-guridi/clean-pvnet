import os
import shutil

from plyfile import PlyData
import numpy as np
from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import tqdm
from PIL import Image
from lib.utils import base_utils
import json
import trimesh
import cv2


def transform_obj_to_ply(model_path):
    # added function to create a ply mesh from an obj mesh
    print("Creating PLY file in ASCII format")
    new_path = model_path
    model_path = model_path[:-3] + "obj"
    mesh = trimesh.load(model_path)
    result = trimesh.exchange.ply.export_ply(mesh, encoding='ascii')
    output_file = open(new_path, "wb+")
    output_file.write(result)
    output_file.close()
    print("PLY file successfully created")


def run_all_custom(data_root):
    # function to run all added custom functions to prepare the data before training
    create_polarized_pics(data_root, "/home/arturo/renders/cup/mitsuba_cup/output/")
    resize_all_images(data_root)


def resize_all_images(data_root):
    rgb_images = os.path.join(data_root, "rgb/")
    masks = os.path.join(data_root, "mask/")
    pol_images = os.path.join(data_root, "pol/")
    camera_intrinsics = os.path.join(data_root, "camera.txt")
    all_images = sorted(os.listdir(rgb_images))
    all_masks = sorted(os.listdir(masks))
    all_polarization = sorted(os.listdir(pol_images))
    assert len(all_images) == len(all_masks), "Error, the len of all the images should be the same as the masks"
    print("Resizing all images, masks and camera")
    width = 0
    height = 0
    width_ratio = 0.0
    height_ratio = 0.0
    for image, mask in tqdm.tqdm(zip(all_images, all_masks)):
        # first we resize all the RGB images
        im_path = os.path.join(rgb_images, image)
        img = cv2.imread(im_path)
        width = int(img.shape[1] - img.shape[1] % 32)
        height = int(img.shape[0] - img.shape[0] % 32)
        width = min(width, height)
        height = min(width, height)

        width_ratio = width / img.shape[1]
        height_ratio = height / img.shape[0]
        # if width_ratio == 1 and height_ratio == 1:
        #     print("No resizing is needed")
        #     return
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(im_path, img)

        # secondly we resize all the masks to the same shape as the images
        mask_path = os.path.join(masks, mask)
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(mask_path, mask)

    # thirdly, we reshape on a separate loop all the polarized images ( because their length is not the same as RGB
    # or mask )
    for pol in all_polarization:
        pol_path = os.path.join(pol_images, pol)
        pol_img = cv2.imread(pol_path)
        pol_img = cv2.resize(pol_img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(pol_path, pol_img)

    # lastly we reshape the cameras intrinsics
    with open(camera_intrinsics, "r") as camera:
        intrinsics = camera.readlines()
    K = np.zeros(shape=(3, 3))
    for i, line in enumerate(intrinsics):
        K[i, :] = np.fromstring(line, dtype=float, sep=" ")
    K[0, :] *= width_ratio
    K[1, :] *= height_ratio
    K = K.flatten().tolist()
    K_str = ""
    for i, intrinsic in enumerate(K):
        if i == 2 or i == 5:
            K_str += str(intrinsic).zfill(16) + "\n"
        else:
            K_str += str(intrinsic).zfill(16) + " "
    with open(camera_intrinsics, "w") as camera:
        camera.write(K_str)
    print(f"All images, mask and camera intrinsics have been resized to {width}x{height}")


def create_polarized_pics(old_data_root, source_image_path):
    # Copies the specified polarized images into the training path
    # note, source_images should be the folder output from the rendering of mitsuba
    # in it, 200 folders, one for each pose, and inside the folder all the images
    print("Copying all images to a new polarization folder")
    pol_path = os.path.join(old_data_root, "pol/")

    source_images_type = ["stokes_s1.jpg", "stokes_s2.jpg"]
    # source_images_type = ["stokes_dolp.jpg", "stokes_aolp.jpg"]
    source_images = sorted(os.listdir(source_image_path))
    source_images.remove("lava")        # this was a test folder not used anymore
    if not os.path.isdir(pol_path):
        os.mkdir(pol_path)
    elif os.path.isdir(pol_path) and len(os.listdir(pol_path)) > 0:
        print("All the images have already been copied")

    for folder_image in source_images:
        base_path = os.path.join(source_image_path, folder_image)
        for image_type in source_images_type:
            image_org = os.path.join(base_path, image_type)
            dest = os.path.join(pol_path, str(folder_image) + image_type.strip("stokes"))
            shutil.copy2(image_org, dest)

    print("All the images were successfully copied")


def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points


def sample_fps_points(data_root):
    ply_path = os.path.join(data_root, 'model.ply')
    if not os.path.isfile(ply_path):
        transform_obj_to_ply(ply_path)

    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def record_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    pose_dir = os.path.join(data_root, 'pose')
    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask')

    inds = range(len(os.listdir(rgb_dir)))

    for ind in tqdm.tqdm(inds):
        rgb_path = os.path.join(rgb_dir, '{}.jpg'.format(ind))

        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        pose = np.load(pose_path)
        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)[0]
        fps_2d = base_utils.project(fps_3d, K, pose)

        mask_path = os.path.join(mask_dir, '{}.png'.format(ind))

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'real', 'cls': 'cat'})
        annotations.append(anno)

    return img_id, ann_id


def custom_to_coco(data_root):
    model_path = os.path.join(data_root, 'model.ply')

    renderer = OpenGLRenderer(model_path)
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.loadtxt(os.path.join(data_root, 'fps.txt'))

    model_meta = {
        'K': K,
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations)
    categories = [{'supercategory': 'none', 'id': 1, 'name': 'cat'}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)
