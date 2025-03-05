from direct3d.pipeline import Direct3dPipeline

pipeline = Direct3dPipeline.from_pretrained("DreamTechAI/Direct3D")
pipeline.to("cuda")

mesh = pipeline(
    "assets/rock.jpg",
    remove_background=False, # set to True if the background of the image needs to be removed
    mc_threshold=-1.0,
    guidance_scale=4.0,
    num_inference_steps=50,
)["meshes"][0]
mesh.export("output.obj")