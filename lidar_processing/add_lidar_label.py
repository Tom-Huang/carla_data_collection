import os
import numpy as np
import plyfile as plyf
import npy2png

data_folder = "./data/for_training"

WINDOW_HEIGHT = 960
WINDOW_WIDTH = 960
FOV = 90

semantic_dict = {}
semantic_dict[0] = (0, 0, 0)  # not labeled
semantic_dict[1] = (255, 255, 0)  # building 70 70 70
semantic_dict[2] = (190, 153, 153)  # fence
semantic_dict[3] = (250, 170, 160)  # other
semantic_dict[4] = (220, 20, 60)  # pedestrain
semantic_dict[5] = (153, 153, 153)  # pole
semantic_dict[6] = (157, 234, 50)  # road line
semantic_dict[7] = (128, 64, 128)  # road
semantic_dict[8] = (244, 35, 232)  # side walk
semantic_dict[9] = (107, 142, 35)  # vegetation
semantic_dict[10] = (0, 0, 142)  # car
semantic_dict[11] = (102, 102, 156)  # wall
semantic_dict[12] = (220, 220, 0)  # traffic sign


def find_lidar_data_path(foldername: str):
    dir_list = os.listdir(foldername)
    data_paths = []
    data_paths.append(foldername + "/" + dir_list[0] + "/" + "Lidar/")
    return data_paths


def get_intrinsic(image_size_x: int, image_size_y: int, camera_fov: float):
    """
    :param image_size_x: the width of image in pixel
    :param image_size_y: the height of image in pixel
    :param camera_fov: the field of view in degree
    :return: intrinsics numpy matrix
    """
    f = image_size_x / (2 * np.tan(camera_fov * np.pi / float(360)))
    cx = image_size_x / 2
    cy = image_size_y / 2
    intrinsics = np.zeros((3, 3))
    intrinsics[0, 0] = f
    intrinsics[1, 1] = f
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    intrinsics[2, 2] = 1.0
    return intrinsics


def projection(p_3d, intrinsics):
    p_2d = intrinsics @ p_3d
    p_2d = p_2d / float(p_2d[2])
    if p_2d[0] - np.floor(p_2d[0]) > 0.5:
        p_2d[0] = np.ceil(p_2d[0])
    else:
        p_2d[0] = np.floor(p_2d[0])
    if p_2d[1] - np.floor(p_2d[1]) > 0.5:
        p_2d[1] = np.ceil(p_2d[1])
    else:
        p_2d[1] = np.floor(p_2d[1])
    # ########################
    # print(p_3d)
    # print(p_2d)
    # ########################
    return int(p_2d[0]),int(p_2d[1])


class LidarData:
    def __init__(self, data_path, data_filename):
        if data_path[-1] != "/":
            data_path = data_path + "/"
        self.data_path = data_path
        self.data_filename = data_filename
        self.ply_data = plyf.PlyData.read(data_path + data_filename)
        self.ply_left_data_ids = []
        self.ply_front_data_ids = []
        self.ply_right_data_ids = []
        self.ply_rear_data_ids = []
        self.tag_left = []
        self.tag_right = []
        self.tag_front = []
        self.tag_rear = []
        rad_deg = np.pi / float(180)
        rad_45 = 45 * rad_deg
        rad_135 = 135 * rad_deg
        rad_180 = np.pi
        rad_225 = 225 * rad_deg
        rad_315 = 315 * rad_deg

        for i, p_3d in enumerate(self.ply_data.elements[0].data):
            rad = np.arctan2(-p_3d[1], -p_3d[0])
            if rad_45 <= rad < rad_135:
                self.ply_front_data_ids.append(i)
            elif rad_135 <= rad < rad_180 or -rad_180 <= rad < -rad_135:
                self.ply_right_data_ids.append(i)
            elif -rad_135 <= rad < -rad_45:
                self.ply_rear_data_ids.append(i)
            elif -rad_45 <= rad < rad_45:
                self.ply_left_data_ids.append(i)

    def add_tag(self, seg_image_data, direction, image_size_x: int, image_size_y: int, camera_fov: float):
        h, w, d = seg_image_data.shape
        if direction == 'front':
            point_inds = self.ply_front_data_ids
        elif direction == 'left':
            point_inds = self.ply_left_data_ids
        elif direction == 'right':
            point_inds = self.ply_right_data_ids
        elif direction == 'rear':
            point_inds = self.ply_rear_data_ids
        else:
            raise Exception("direction argument can only be left, right, rear, front!")
        intrinsic_mat = get_intrinsic(image_size_x, image_size_y, camera_fov)
        tags = []
        #############################
        unique, counts = np.unique(seg_image_data, return_counts=True)
        print(dict(zip(unique, counts)))
        outliers_num = 0
        inliers_num = 0
        #############################
        for p_ind in point_inds:
            p_3d = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            if direction == 'front':
                p_3d[0] = self.ply_data.elements[0].data[p_ind][0]
                p_3d[1] = self.ply_data.elements[0].data[p_ind][2]
                p_3d[2] = -self.ply_data.elements[0].data[p_ind][1]
            elif direction == 'left':
                p_3d[0] = -self.ply_data.elements[0].data[p_ind][1]
                p_3d[1] = self.ply_data.elements[0].data[p_ind][2]
                p_3d[2] = -self.ply_data.elements[0].data[p_ind][0]
            elif direction == 'right':
                p_3d[0] = self.ply_data.elements[0].data[p_ind][1]
                p_3d[1] = self.ply_data.elements[0].data[p_ind][2]
                p_3d[2] = self.ply_data.elements[0].data[p_ind][0]
            elif direction == 'rear':
                p_3d[0] = -self.ply_data.elements[0].data[p_ind][0]
                p_3d[1] = self.ply_data.elements[0].data[p_ind][2]
                p_3d[2] = self.ply_data.elements[0].data[p_ind][1]
            p_2d = projection(p_3d.reshape((3, 1)), intrinsic_mat)

            if p_2d[0] < 0 or p_2d[0] >= image_size_x or p_2d[1] < 0 or p_2d[1] >= image_size_y:
                tags.append(0)
                outliers_num += 1
            else:
                tags.append(seg_image_data[p_2d[1], p_2d[0]][0])
                inliers_num += 1

            if "030" in self.data_filename and (-18 < self.ply_data.elements[0].data[p_ind][0] < -13) and (-2 < self.ply_data.elements[0].data[p_ind][1] < -0.5):
                p_2d = projection(p_3d.reshape((3, 1)), intrinsic_mat)
                print(intrinsic_mat)
                print(self.ply_data.elements[0].data[p_ind], "\t", p_2d, "\t", seg_image_data[p_2d[1], p_2d[0]][0])


        ############################
        print("outliers num: ", outliers_num)
        print("inliers num: ", inliers_num)
        ############################

        if direction == 'front':
            self.tag_front = tags
        elif direction == 'left':
            self.tag_left = tags
        elif direction == 'right':
            self.tag_right = tags
        elif direction == 'rear':
            self.tag_rear = tags

    def save_labeled_data_as_ply(self, save_path):
        labeled_data_filename = "labeled_" + self.data_filename
        vertex = np.empty((len(self.ply_data.elements[0].data)),
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),('label','u1')])
        vertex_i = 0
        for i, p_i in enumerate(self.ply_front_data_ids):
            p_3d = self.ply_data.elements[0].data[p_i]
            vertex[vertex_i] = (
                p_3d[0], p_3d[1], p_3d[2], semantic_dict[self.tag_front[i]][0], semantic_dict[self.tag_front[i]][1],
                semantic_dict[self.tag_front[i]][2], self.tag_front[i])
            vertex_i += 1

        for i, p_i in enumerate(self.ply_left_data_ids):
            p_3d = self.ply_data.elements[0].data[p_i]
            vertex[vertex_i] = (
                p_3d[0], p_3d[1], p_3d[2], semantic_dict[self.tag_left[i]][0], semantic_dict[self.tag_left[i]][1],
                semantic_dict[self.tag_left[i]][2], self.tag_left[i])
            vertex_i += 1

        for i, p_i in enumerate(self.ply_right_data_ids):
            p_3d = self.ply_data.elements[0].data[p_i]
            vertex[vertex_i] = (
                p_3d[0], p_3d[1], p_3d[2], semantic_dict[self.tag_right[i]][0], semantic_dict[self.tag_right[i]][1],
                semantic_dict[self.tag_right[i]][2], self.tag_right[i])
            vertex_i += 1

        for i, p_i in enumerate(self.ply_rear_data_ids):
            p_3d = self.ply_data.elements[0].data[p_i]
            vertex[vertex_i] = (
                p_3d[0], p_3d[1], p_3d[2], semantic_dict[self.tag_rear[i]][0], semantic_dict[self.tag_rear[i]][1],
                semantic_dict[self.tag_rear[i]][2], self.tag_rear[i])
            vertex_i += 1

        el = plyf.PlyElement.describe(vertex, 'vertex')
        plyf.PlyData([el]).write(save_path + labeled_data_filename)


if __name__ == "__main__":
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    lidar_data_path = find_lidar_data_path(data_folder)
    lidar_data_filename_list = os.listdir(lidar_data_path[0])
    for lidar_filename in lidar_data_filename_list:
        lidar_data = LidarData(lidar_data_path[0], lidar_filename)
        seg_filename, seg_img_ind = lidar_filename.rstrip(".ply")[0], int(lidar_filename.rstrip(".ply")[1:]) - 1
        seg_filename = seg_filename + ".npy"

        seg_data_paths = npy2png.find_seg_data_path(data_folder)
        for seg_data_path in seg_data_paths:
            if seg_data_path[-1] != "/":
                seg_data_path = seg_data_path + "/"
            direction_flag = ""
            if "left" in seg_data_path:
                direction_flag = "left"
            elif "right" in seg_data_path:
                direction_flag = "right"
            elif "front" in seg_data_path:
                direction_flag = "front"
            elif "rear" in seg_data_path:
                direction_flag = "rear"
            else:
                continue
            seg_img = np.load(seg_data_path + seg_filename)
            lidar_data.add_tag(seg_img[seg_img_ind], direction_flag, WINDOW_WIDTH, WINDOW_HEIGHT, FOV)
        print("tags for {0} added.".format(lidar_filename))
        lidar_save_path = lidar_data_path[0]
        if lidar_save_path[-1] != "/":
            lidar_save_path = lidar_save_path + "/"
        lidar_save_path = lidar_save_path + "labeled/"
        if not os.path.exists(lidar_save_path):
            os.makedirs(lidar_save_path)

        lidar_data.save_labeled_data_as_ply(lidar_save_path)
        # next step add label int to lidar ply file
