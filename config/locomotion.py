# import socket
# 这个函数可能用于实验设置、数据分析、测试调节等场景, 用于生成对应设置参数清晰的标识名。
from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ## value kwargs
    ("discount", "d"),
]

logbase = "logs"

base = {
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 32,
        "n_diffusion_steps": 20,
        "action_weight": 10,
        "loss_weights": None,
        "loss_discount": 0.97,
        "predict_epsilon": False,
        "dim_mults": (1, 2, 4, 8),
        "attention": False,
        "renderer": "utils.MatplotRenderer",
        ## dataset
        "loader": "datasets.SequenceDataset",
        "normalizer": "GaussianNormalizer",  # GaussianNormalizer LimitsNormalizer
        "preprocess_fns": [],
        "clip_denoised": False,
        "use_padding": False,
        "max_path_length": 1000,
        ## serialization
        "logbase": logbase,
        "prefix": "diffusion/test",
        "exp_name": watch(args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 5e4,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.97,
        "save_freq": 5000,
        "sample_freq": 2000,
        "n_saves": 5,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,
    },
    "values": {
        "model": "models.ValueFunction",
        "diffusion": "models.ValueDiffusion",
        "horizon": 32,
        "n_diffusion_steps": 20,
        "dim_mults": (1, 2, 4, 8),
        "renderer": "utils.MatplotRenderer",
        ## value-specific kwargs
        "discount": 0.97,  # 0.99,
        "termination_penalty": -100,
        "normed": False,
        ## dataset
        "loader": "datasets.ValueDataset",
        "normalizer": "GaussianNormalizer",  # GaussianNormalizer LimitsNormalizer
        "preprocess_fns": [],
        "use_padding": False,
        "max_path_length": 1000,
        ## serialization
        "logbase": logbase,
        "prefix": "values/combined_reward",
        "exp_name": watch(args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "value_l2",
        "n_train_steps": 5e4,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.98,
        "save_freq": 1000,
        "sample_freq": 0,
        "n_saves": 5,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,
    },
    "plan": {
        "guide": "sampling.ValueGuide",
        "policy": "sampling.GuidedPolicy",
        "max_episode_length": 200,
        "batch_size": 64,
        "preprocess_fns": [],
        "device": "cuda",
        "seed": None,
        ## sample_kwargs
        "n_guide_steps": 2,
        "scale": 0.1,
        "t_stopgrad": 2,
        "scale_grad_by_std": True,
        ## serialization
        "loadbase": None,
        "logbase": logbase,
        "prefix": "plans/",
        "exp_name": watch(args_to_watch),
        "vis_freq": 10,
        "max_render": 8,
        ## diffusion model
        "horizon": 32,
        "n_diffusion_steps": 20,
        ## value function
        "discount": 0.97,
        ## loading
        "diffusion_loadpath": "f:diffusion/hird_g_H{horizon}_T{n_diffusion_steps}",
        "value_loadpath": "f:values/combined_reward_H{horizon}_T{n_diffusion_steps}_d{discount}",
        "diffusion_epoch": "latest",
        "value_epoch": "latest",
        "verbose": True,
        "suffix": "2",
    },
}


# ------------------------ overrides ------------------------#


# hopper_medium_expert_v2 = {
#     "plan": {
#         "scale": 0.0001,
#         "t_stopgrad": 4,
#     },
# }


# halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
#     "diffusion": {
#         "horizon": 4,
#         "dim_mults": (1, 4, 8),
#         "attention": True,
#     },
#     "values": {
#         "horizon": 4,
#         "dim_mults": (1, 4, 8),
#     },
#     "plan": {
#         "horizon": 4,
#         "scale": 0.001,
#         "t_stopgrad": 4,
#     },
# }
