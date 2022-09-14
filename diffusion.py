from ast import Bytes
import base64
from torch import autocast
import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionInpaintPipeline
import PIL

class Diffusion:
    device = "cuda"
    model_id_or_path = "CompVis/stable-diffusion-v1-4"

    def load_img(self, img : str, h0 : int, w0 : int):
        image = Image.open(BytesIO(base64.b64decode(img))).convert("RGB")

        w, h = image.size

        if h0 is not None and w0 is not None:
            h, w = h0, w0

        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)

        return image

    def image_to_base64(self, img, format):
        buffered = BytesIO()
        img.save(buffered, format=format)
        return str(base64.b64encode(buffered.getvalue()).decode("utf-8"))


    def image(self, prompt : str, image : str, format : str = "JPEG"):

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id_or_path,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=True
        )

        pipe = pipe.to(self.device)

        init_image = self.load_img(image,512,512)

        with autocast("cuda"):
            images = pipe(
                prompt, 
                init_image=init_image, 
                strength=0.75, 
                guidance_scale=7.5
            ).images

            return [self.image_to_base64(img,format) for img in images]


    def inpaint(self,prompt : str, img : str,mask : str, format : str = "JPEG"):

        #img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        #mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

        init_image = self.load_img(img,512,512)
        mask_image = self.load_img(mask,512,512)

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id_or_path,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=True
        )

        pipe = pipe.to(self.device)

        with autocast("cuda"):
            images = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75).images
            return [self.image_to_base64(img, format) for img in images]

    def text(self, prompt : str, format : str = "JPEG"):
        pipe = StableDiffusionPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)
        pipe = pipe.to(self.device)

        with autocast("cuda"):
            images = pipe(prompt, guidance_scale=7.5).images
            return [self.image_to_base64(img, format) for img in images]
