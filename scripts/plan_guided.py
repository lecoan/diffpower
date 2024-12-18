# import pdb
# 导入一个名为 diffuser.utils的Python模块，并将其重命名为﻿utils以便在后续的代码中使用。
import diffuser.utils as utils
from diffuser.environments.power import PowerEnv
import diffuser.sampling as sampling

# python scripts/plan_guided.py --dataset power --logbase logs/pretrained


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = "walker2d-medium-replay-v2"
    config: str = "config.locomotion"

args = Parser().parse_args("plan")


# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase,
    args.dataset,
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch,
    seed=args.seed,
    device=args.device,
)
value_experiment = utils.load_diffusion(
    args.loadbase,
    args.dataset,
    args.value_loadpath,
    epoch=args.value_epoch,
    seed=args.seed,
    device=args.device,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

# ema：滑动平均模型
# 这是在读取模型
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
# 这是在读取模型
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

# env = dataset.env
env = PowerEnv()
observation = env.reset()
renderer.set_env(env)

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(args.max_episode_length):

    if t % 10 == 0:
        print(args.savepath, flush=True)

    ## save state for rendering only
    state = env.state_vector().copy()

    ## format current observation for conditioning
    conditions = {0: observation}
    actions, samples = policy(
        conditions, batch_size=args.batch_size, verbose=args.verbose
    )
    # print(actions)

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(actions.tolist())

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
        f"values: {samples.values.sum()} | scale: {args.scale}",
        flush=True,
    )

    ## update rollout observations
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation

## write results to json file at `args.savepath`
logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
