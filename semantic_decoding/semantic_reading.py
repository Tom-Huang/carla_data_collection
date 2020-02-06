import numpy as np
from PIL import Image
import os
from npy2png import find_data_filename

WINDOW_HEIGHT = 375
WINDOW_WIDTH = 1242
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

image_path = "./2019-07-10_15_53/semantic/006012.png"
image_folder = "./semantic/"
save_folder = "./processed_semantic/"

def semantic_image_generator(raw_data):
    # this function convert raw semantic data to rgb image with different colors representing different objects
    # raw_data is array of size W X H, each element stores an integer corresponding to a certain class of object
    # return a W X H np.array type
    h, w = raw_data.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    print(w, h)
    for r in range(h):
        for c in range(w):
            for i in range(3):
                # print(r,c,i)
                # print(raw_data[r, c])
                output[r, c, i] = semantic_dict[int(raw_data[r, c])][i]
    return output

if __name__ == "__main__":

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_list = find_data_filename(image_folder)
    for image_name in image_list:
        im = Image.open(image_folder+image_name)
        im_array = np.array(im)[:,:,0]
        rgb_img_np = semantic_image_generator(im_array)
        rgb_img_pil = Image.fromarray(rgb_img_np)
        rgb_img_pil.save(save_folder+image_name)
