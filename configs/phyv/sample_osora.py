image_size = (256,256)
num_frames = 334
fps = 20
frame_interval = 1
save_fps = 20

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5
# prompt = ["276"]
# reference_path = ["/local2/xingcheng/Open-Sora/debug/condition/00101-006-0.jpg"]
# prompt = ["23"]
# reference_path = ["/local2/xingcheng/Open-Sora/debug/condition/00023-875-18.jpg"]
prompt = ["In two dimension, one red ball and one green ball free fall and bumping and colliding, following Newton's law of dynamics. camera static; single consistent scene; white background;"]
# prompt = ["Beautiful sunset, purple sky, palm trees"]
#reference_path = ["/local2/xingcheng/Open-Sora/debug/condition/00007-008-7.jpg"]
#mask_strategy = ["0"]
reference_path = ["/local2/xingcheng/data/phyre/ball_cross_template/test/00007:008/7-f.mp4"]
mask_strategy = ["0,0,0,0,20,0"]
sample_name = "first-10-cond-open-sora-p4-20"

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = None
flow = None