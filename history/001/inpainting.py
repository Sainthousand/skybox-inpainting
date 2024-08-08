import torch
from diffusers import AutoPipelineForInpainting, StableDiffusionXLPipeline
from diffusers.utils import load_image, make_image_grid
import PIL.Image
from PIL import Image, ImageOps
import numpy as np
import os
from numba import jit, vectorize, int64


def generate_image(init_image, mask_image, output, idx_str):
    pipeline = AutoPipelineForInpainting.from_pretrained(
    # pipeline = StableDiffusionXLPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16, 
        variant="fp16",
        # custom_pipeline="pipeline_stable_diffusion_xl_differential_img2img"
    ).to("cuda:0")
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()

    generator = torch.Generator("cuda:0").manual_seed(34890123)
    
    prompt = "A high-definition image of real sky, blue sky, with some clouds"
    negative_prompt = "word, slogan, ground, tree, blurred, ceiling, branches low quality, pixelated, Disfigured, cartoon, nude, building, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,"
    mask_blurred = pipeline.mask_processor.blur(mask_image, blur_factor=20)
    image = pipeline(prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image, 
                    mask_image=mask_blurred, 
                    generator=generator, 
                    # target_size=(2048, 4096), 
                    # height=2048, 
                    # width=4096, 
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    strenght=1.0).images[0]
    result = make_image_grid([init_image, mask_blurred, image], rows=3, cols=1)

    width, height = image.size
    quarter_height = height // 4
    area = (0, quarter_height*2, width, height)
    cropped_img = image.crop(area)
    resized_img = cropped_img.resize((4096, 2048), Image.BILINEAR)

    resized_img.save(os.path.join(output_path, f'{idx_str}.jpg'))
    result.save(os.path.join('/home/shengqian.li/code/skybox_inpainting/result', f'{idx_str}.jpg'))


def scale_and_paste(original_image, mask_image=None, resize_method=Image.NEAREST):
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = 1024
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = 1024
        new_width = round(new_height * aspect_ratio)

    wb_img = Image.new("RGB", (original_image.width, original_image.height), "white")
    if mask_image is not None: 
        wb_img.paste(original_image, (0, 0), mask_image)
    else:
        wb_img.paste(original_image, (0, 0))
    wb_img = wb_img.resize((new_width, new_height), resize_method)
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


if __name__=="__main__":
    np.random.seed(24)
    for idx in range(1, 932):
        # path
        root_path = "/home/shengqian.li/code/skybox_inpainting"
        idx_str = str(idx).zfill(6)
        img_dir_path = f'{root_path}/frame'
        mask_dir_path = f'{root_path}/mask'
        output_path=f'{root_path}/output'
        img_path = os.path.join(img_dir_path, f'{idx_str}.jpg')
        mask_path = os.path.join(mask_dir_path, f'{idx_str}.png')
        # load image
        init_image = load_image(img_path)
        mask_image = load_image(mask_path)
        # np_mask = np.logical_not(np.array(mask_image))
        inverted_mask = Image.eval(mask_image.convert("1"), lambda x: 255 - x)
        wb_img = scale_and_paste(init_image, inverted_mask, Image.BILINEAR)
        wb_mask = scale_and_paste(mask_image)
        # wb_img = refine_back_pixel(wb_img, wb_mask)
        # wb_img = add_gauss_noise_to_masked_image(wb_img, wb_mask)
        wb_img = fullfill_image_from_mask(wb_img, wb_mask)

        white_image = Image.new('RGB', (wb_img.width, wb_img.height), (255, 255, 255))
        noise = np.random.normal(0, 20, np.array(wb_img).shape).astype(np.uint8)
        noisy_image_pil = Image.fromarray(noise)

        generate_image(wb_img, wb_mask, output_path, idx_str)

        