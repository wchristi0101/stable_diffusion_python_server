# stable_diffusion_python_server
A simple python server for locally working with stable diffusion

To setup : 
- Have python 3.7 or greater installed.
- Make sure you are using a Nvidia GPU, otherwise the diffusion class will need some updates
- Reccomended documentation for Apple Silicon M1 / M2 for edits : https://huggingface.co/docs/diffusers/optimization/mps
- Create a hugging face account, and then accept the (license)[https://huggingface.co/CompVis/stable-diffusion-v1-4]
- Generate an access token. For more information visit (here) https://huggingface.co/docs/hub/security-tokens
- Install packages with :  ```pip install -r requirements.txt```
    - Note : If you are not using a Nvidia GPU you may need to choose a different version of torch / torchvision.
- Run the following : ```huggingface-cli login```
- Run ```python main.py```
- You can use the postman collection as a starting point.
