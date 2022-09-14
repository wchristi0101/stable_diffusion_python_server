from sanic import Sanic, json
from diffusion import Diffusion

app = Sanic("SimpleAPI")
diffusion = Diffusion()

HOST = "localhost"
PORT = 8000

@app.post("/texttoimage")
async def text_to_image(request, *args, **kwargs):
    prompt = request.json.get("prompt")

    if(prompt == ""): return app.exception("prompt may not be empty", status_code="401")
    if(type(prompt) != str): return app.exception("prompt must be a string", status_code="401")

    return json( {"images" : diffusion.text(prompt)})

@app.post("/imagetoimage")
async def image_to_image(request, *args, **kwargs):
    prompt = request.json.get("prompt")
    image = request.json.get("image")

    if(prompt == ""): return app.exception("prompt may not be empty", status_code="401")
    if(type(prompt) != str): return app.exception("prompt must be a string", status_code="401")

    if(image == ""): return app.exception("image may not be empty", status_code="401")
    if(type(image) != str): return app.exception("image must be a base64 string", status_code="401")

    return json({"images" : diffusion.image(prompt,image)})

@app.post("inpaint")
async def inpaint(request, *args, **kwargs):
    prompt = request.json.get("prompt")
    image = request.json.get("image")
    mask = request.json.get("mask")

    if(prompt == ""): return app.exception("prompt may not be empty", status_code="401")
    if(type(prompt) != str): return app.exception("prompt must be a string", status_code="401")

    if(image == ""): return app.exception("image may not be empty", status_code="401")
    if(type(image) != str): return app.exception("image must be a base64 string", status_code="401")

    if(mask == ""): return app.exception("mask may not be empty", status_code="401")
    if(type(mask) != str): return app.exception("mask must be a base64 string", status_code="401")

    return json({"images" : diffusion.inpaint(prompt,image,mask)})

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)