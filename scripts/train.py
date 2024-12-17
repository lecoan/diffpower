# 导入一个名为 diffuser.utils的Python模块，并将其重命名为utils以便在后续的代码中使用。
import diffuser.utils as utils
# from diffuser.utils.arrays import set_device



#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#


# 这是定义了一个名为 Parser 的类，该类继承了 utils 模块（某个文件内）中的 Parser 类。
class Parser(utils.Parser):
    # 运行时 python scripts/train.py --dataset power，替换了原来的 “walker2d”
    dataset: str = "walker2d-medium-replay-v2"
    # 在这个 Parser 类中，定义了一个类变量 config，并给它赋值为 'config.locomotion'。
    config: str = 'config.locomotion'

# 要读参数 config.locomotion.diffusion
args = Parser().parse_args('diffusion')


# set_device(args.device)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    # Config/locomotion中的参数，数据加载具体在diffuser.datasets.sequence中处理的
    args.loader,
    # 保存一下设置
    savepath=(args.savepath, 'dataset_config.pkl'),
    #  --dataset power
    env=args.dataset,
    # 数据的 horizon？
    horizon=args.horizon,
    # 归一化。对数据高斯归一化
    normalizer=args.normalizer,
    # 没用。这些预处理函数可能在数据加载进内存前对其进行处理。这里没处理
    preprocess_fns=args.preprocess_fns,
    # 没用。用 padding（填充）方法来使序列长度一致。
    use_padding=args.use_padding,
    # 一条数据的轨迹有多长（一条轨迹采样几次
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# class本身只是创造概念，只有括号之后才会创建一个实例，config()实例：
# 作用在class上的()是init，作用在实例上的()是call
model_config = utils.Config(
    # TemporalUnet
    args.model,
    # 为了后续 plan 调用
    savepath=(args.savepath, 'model_config.pkl'),
    # 实际上是预测 32 步
    horizon=args.horizon,
    # 状态变量+动作
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    # 用于某种方式的维数扩展或多尺度处理。
    dim_mults=args.dim_mults,
    # 没用。
    attention=args.attention,
    # GPU
    device=args.device,
)

diffusion_config = utils.Config(
    # base.diffusion下面的一些
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    # 32？怎么又来一个？这有必要吗又来一遍，还是说会覆盖？
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    # 20？
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    # 没用。是否应该对去噪数据进行剪裁。args 应包含此值。
    clip_denoised=args.clip_denoised,
    # 控制在对扩散模型进行训练时，是否预测用于扩散过程的噪声水平。
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    # 在计算损失时，行动的重要性。
    action_weight=args.action_weight,
    # 没用。可能有助于通过对不同目标应用不等的权重来平衡不同的目标
    loss_weights=args.loss_weights,
    # 可以降低未来损失的重要性。
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    # 类
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    # 多少数据一组训练，得到一个梯度
    train_batch_size=args.batch_size,
    # 模型的学习速率，也就是每次更新参数时的步长。
    train_lr=args.learning_rate,
    # 应该是每隔多少步进行一次梯度累积。
    gradient_accumulate_every=args.gradient_accumulate_every,
    # 是指数加权移动平均（Exponential Moving Average）的衰减速率。
    ema_decay=args.ema_decay,
    # 样本频率和保存频率，可能表示每隔多少步进行一次样本抽样和模型保存。
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    # 标签频率，应该是每隔多少步更新一次标签。什么标签？
    label_freq=int(args.n_train_steps // args.n_saves),
    # 是否并行保存。
    save_parallel=args.save_parallel,
    # 保存结果的目录。
    results_folder=args.savepath,
    # 用于存储数据或结果的存储桶？
    bucket=args.bucket,
    # 可能是引用的数量？
    n_reference=args.n_reference
)


#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, renderer, device=args.device)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

# 报告给定模型的参数统计信息，包括模型中的总参数数量，以及每个模块参数数量排名前十的模块名、模块详细信息及其参数数量。
# 对于编号超过十的模块，函数只报告了这部分模块的总参数数量。
# utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
# dataset中取出一个batch，【中括号调用的是__getitem__】，括号是call，把他转移到cuda设备上
# 只是测试看看能不能跑 -^-
batch = utils.batchify(dataset[0], args.device)
# *batch 解包为：trajectories, conditions
# loss, _ = diffusion.loss(trajectories, conditions)
loss, _ = diffusion.loss(*batch)
# 自动反向传播
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

# 数据集从头训练几次
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)
