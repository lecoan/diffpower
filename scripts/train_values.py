# 导入一个名为 diffuser.utils的Python模块，并将其重命名为utils以便在后续的代码中使用。
import diffuser.utils as utils
# import pdb
# from diffuser.utils.arrays import set_device


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = "walker2d-medium-replay-v2"
    config: str = "config.locomotion"


args = Parser().parse_args("values")

# set_device(args.device)

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,
    # 决定是否会用normalizer
    normed=args.normed,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    ## value-specific kwargs
    discount=args.discount,
    termination_penalty=args.termination_penalty,

)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, renderer, device=args.device)

# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

print("Testing forward...", end=" ", flush=True)
# [] 是 __getitem__
batch = utils.batchify(dataset[0], args.device)

loss, _ = diffusion.loss(*batch)
loss.backward()
print("✓")

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")
    trainer.train(n_train_steps=args.n_steps_per_epoch)
