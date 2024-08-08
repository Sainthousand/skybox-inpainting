import torch
from diffusers import AutoPipelineForInpainting, StableDiffusionXLPipeline
from diffusers.utils import load_image, make_image_grid
import PIL.Image
from PIL import Image, ImageOps
import numpy as np
import os
from numba import jit, vectorize, int64
from tqdm import tqdm
import cv2


def generate_image_sequencce(init_image_list, mask_image_list, root_path):
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16, 
        variant="fp16",
    ).to("cuda:0")
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator("cuda:0").manual_seed(34890123)
    
    prompt = "(sky), real, some clouds"
    negative_prompt = "(comic), (cartoon), signature, text, word, slogan, ground, tree, blurred, ceiling, branches low quality, pixelated, Disfigured, nude, building, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, username, blurry,"

    for idx, (init_image, mask_image) in tqdm(enumerate(zip(init_image_list, mask_image_list))):
        idx_str = str(idx + 1).zfill(6)
        mask_blurred = pipeline.mask_processor.blur(mask_image, blur_factor=5)
        image = pipeline(prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=init_image, 
                        mask_image=mask_blurred, 
                        generator=generator, 
                        num_inference_steps=50,
                        guidance_scale=15.0, # prev: 10.0
                        strenght=0.95).images[0]
        result = make_image_grid([init_image, mask_blurred, image], rows=3, cols=1)

        width, height = image.size
        quarter_height = height // 4
        area = (0, quarter_height*2, width, height)
        cropped_img = image.crop(area)
        resized_img = cropped_img.resize((4096, 2048), Image.BILINEAR)

    resized_img.save(os.path.join(root_path, "output", f'{idx_str}.jpg'))
    result.save(os.path.join(root_path, 'result', f'{idx_str}.jpg'))


def generate_image(init_image, mask_image, root_path, idx_str):
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16, 
        variant="fp16",
    ).to("cuda:0")
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator("cuda:0").manual_seed(34890123)
    
    prompt = "A high-definition image of real sky, with some clouds"
    negative_prompt = "boat, Artifact, tower, error, sketch, horror, geometry, stone, cartoon, word, slogan, ground, tree, blurred, ceiling, branches, pixelated, Disfigured, cartoon, nude, building, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,"
    mask_blurred = pipeline.mask_processor.blur(mask_image, blur_factor=5)

    image = pipeline(prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image, 
                    mask_image=mask_blurred, 
                    generator=generator, 
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    strenght=0.9).images[0]
    result = make_image_grid([init_image, mask_blurred, image], rows=3, cols=1)

    width, height = image.size
    quarter_height = height // 4
    area = (0, quarter_height*2, width, height)
    cropped_img = image.crop(area)
    resized_img = cropped_img.resize((4096, 2048), Image.BILINEAR)

    resized_img.save(os.path.join(root_path, "output", f'{idx_str}.jpg'))
    result.save(os.path.join(root_path, 'result', f'{idx_str}.jpg'))


def scale_and_paste(original_image, mask_image=None, resize_method=Image.NEAREST):
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = 1024
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = 1024
        new_width = round(new_height * aspect_ratio)

    original_image_resized = original_image.resize((new_width, new_height), resize_method)
    wb_img = Image.new("RGB", (new_width, new_height), "white")
    if mask_image is not None: 
        mask_image_resized = mask_image.resize((new_width, new_height), Image.NEAREST)
        wb_img.paste(original_image_resized, (0, 0), mask_image_resized)
    else:
        wb_img.paste(original_image_resized, (0, 0))

    wb_img = ImageOps.expand(wb_img, border=(0, (1024-new_height), 0, 0), fill=(255, 255, 255))

    return wb_img


def add_gauss_noise_to_masked_image(image, mask, noise_level=50):
    np_img = np.array(image)
    np_mask = np.array(mask.convert("L"))
    noise = np.random.normal(0, noise_level, np_img.shape).astype(np.uint8)
    noisy_image = np.where(np_mask[:, :, None] > 0, noise, np_img)
    noisy_image_pil = Image.fromarray(noisy_image)
    return noisy_image_pil


def fullfill_image_from_mask(image, mask):
    np_img = np.array(image)
    np_mask = np.array(mask.convert("L"))
    mask_inside_pixels = np_img[np_mask == 0]
    random_pixels = mask_inside_pixels[np.random.randint(0, len(mask_inside_pixels), size=np_mask.shape)]
    result_image_array = np.where(np_mask[:, :, None] == 255, random_pixels, np_img)
    result_image = Image.fromarray(result_image_array)
    return result_image


def refine_back_pixel(image, mask):
    np_img = np.array(image)
    np_mask = np.array(mask.convert("L"))
    gray_array = np.array(Image.fromarray(np_img).convert("L"))

    # define threshold
    dark_threshold = 150
    light_threshold = 200

    dark_indices = np.where((gray_array < dark_threshold) & (np_mask == 0))
    light_indices = np.where((gray_array > light_threshold) & (np_mask == 0))

    if len(light_indices[0]) > 0:
        random_light_pixels = np_img[light_indices][np.random.choice(len(light_indices[0]), len(dark_indices[0]))]
        result_image_array = np_img.copy()
        result_image_array[dark_indices] = random_light_pixels
    else:
        result_image_array = np_img
    return Image.fromarray(result_image_array)


def refine_mask(mask, root_path, idx_str):
    image_array = np.array(mask.convert("L"))
    height, width = image_array.shape

    # fill the bottom half with white
    white_fill = np.ones((height // 2, width), dtype=np.uint8) * 255
    white_filled_mask = np.vstack((image_array[:height // 2], white_fill))

    # erode and dilate
    eroded = cv2.erode(white_filled_mask, np.ones((5, 5), np.uint8), iterations=1)
    dilatied = cv2.dilate(eroded, np.ones((5, 5), np.uint8), iterations=10)
    new_mask = Image.fromarray(dilatied)
    new_mask.save(f"{root_path}/mask_refined/{idx_str}.png")

    return new_mask


if __name__=="__main__":
    np.random.seed(24)
    root_path = "/home/shengqian.li/code/skybox_inpainting"

    wb_img_list = []
    wb_mask_list = []
    
    for idx in tqdm(range(1, 100)):
        # path
        idx_str = str(idx).zfill(6)
        img_dir_path = f'{root_path}/frame'
        mask_dir_path = f'{root_path}/mask'
        img_path = os.path.join(img_dir_path, f'{idx_str}.jpg')
        mask_path = os.path.join(mask_dir_path, f'{idx_str}.png')
        # load image
        init_image = load_image(img_path)
        mask_image = load_image(mask_path)
        # refine mask and img
        mask_image = refine_mask(mask_image, root_path, idx_str)
        inverted_mask = Image.eval(mask_image.convert("1"), lambda x: 255 - x)
        wb_img = scale_and_paste(init_image, inverted_mask, Image.BILINEAR)
        wb_img.save(f"{root_path}/wb_img/{idx_str}.png")
        wb_mask = scale_and_paste(mask_image)
        wb_img = fullfill_image_from_mask(wb_img, wb_mask)

        wb_img_list.append(wb_img)
        wb_mask_list.append(wb_mask)

    generate_image_sequencce(wb_img_list, wb_mask_list, root_path)

        