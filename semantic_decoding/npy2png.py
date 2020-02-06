import numpy as np
from PIL import Image
import os

WINDOW_HEIGHT = 540
WINDOW_WIDTH = 960
SEMANTIC_FLAG = 1

semantic_dict = {}
semantic_dict[0] = (0, 0, 0)  # not labeled
semantic_dict[1] = (70, 70, 70)  # building
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

data_folder = "./data/for_training"


def semantic_image_generator(raw_data):
    # this function convert raw semantic data to rgb image with different colors representing different objects
    # raw_data is array of size W X H, each element stores an integer corresponding to a certain class of object
    # return a W X H X 3 np.array type
    h, w, _ = raw_data.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    print(w, h)
    for r in range(h):
        for c in range(w):
            for i in range(3):
                # print(r,c,i)
                output[r, c, i] = semantic_dict[int(raw_data[r, c])][i]
    return output


def find_seg_data_path(foldername):
    dir_list = os.listdir(foldername)

    data_paths = []
    data_paths.append(foldername + "/" + dir_list[0] + "/" + "left/CameraSemSeg/")
    data_paths.append(foldername + "/" + dir_list[0] + "/" + "right/CameraSemSeg/")
    data_paths.append(foldername + "/" + dir_list[0] + "/" + "front/CameraSemSeg/")
    data_paths.append(foldername + "/" + dir_list[0] + "/" + "rear/CameraSemSeg/")
    return data_paths


def find_data_filename(data_path):
    filenames = os.listdir(data_path)
    print(filenames)
    return filenames


def save_images(data_path, data_filename, save_path):
    if data_path[-1] != "/":
        data_path = data_path + "/"

    images = np.load(data_path + data_filename)
    for i, image in enumerate(images):
        print(image.shape)
        print("Processing image: %d" % i)
        if SEMANTIC_FLAG:
            image = semantic_image_generator(image)
        img_pil = Image.fromarray(image)

        if save_path[-1] != "/":
            save_path = save_path + "/"
        img_pil.save(save_path + data_filename.rstrip(".npy") + str(i) + ".png")


if __name__ == "__main__":
    data_paths = find_seg_data_path(data_folder)
    for data_path in data_paths:
        data_filename_list = find_data_filename(data_path)
        for data_filename in data_filename_list:
            if data_filename.split(".")[-1] == "npy":
                save_images(data_path, data_filename, data_path)
