import os
from PIL import Image
from tqdm import tqdm


root_path = "/data/nas/main/uniwheel/uw-eval/20240603"
output_path = "/home/shengqian.li/code/skybox_inpainting"
img_output_path = os.path.join(output_path, "frame")
mask_output_path = os.path.join(output_path, "mask")

count = 3
sub_set_name_list = os.listdir(root_path)
for sub_set_name in tqdm(sub_set_name_list):
    dir_path = os.path.join(root_path, sub_set_name, "splatfacto-scene-graph")
    if not os.path.exists(dir_path):
        continue
    data_name_list = os.listdir(dir_path)
    for data_name in data_name_list:
        img_path = os.path.join(dir_path, data_name, "ldr_sky_sphere.jpg")
        mask_path = os.path.join(dir_path, data_name, "ldr_sky_sphere_mask.png")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            Image.open(img_path).save(os.path.join(img_output_path, str(count).zfill(6)  + ".jpg"))
            Image.open(mask_path).save(os.path.join(mask_output_path, str(count).zfill(6)  + ".png"))
            count += 1