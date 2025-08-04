import PIL
import requests
import torch
from io import BytesIO
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import  StableDiffusionPipeline_LEDITS

model = 'runwayml/stable-diffusion-v1-5'
#model = '/workspace/StableDiff/models/stable-diffusion-v1-5'

device = 'cuda'

pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model,safety_checker = None,)
pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(model, subfolder="scheduler"
                                                             , algorithm_type="sde-dpmsolver++", solver_order=2)
pipe.to(device)


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")
gen = torch.Generator(device=device)

gen.manual_seed(21)
img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/cherry_blossom.png"
image = download_image(img_url)
_ = pipe.invert( image = image,
    num_inversion_steps=50,
    skip=0.1
    )
edited_image = pipe(
    editing_prompt=["cherry blossom"],
    edit_guidance_scale=10.0,
    edit_threshold=0.75,
        ).images[0]
