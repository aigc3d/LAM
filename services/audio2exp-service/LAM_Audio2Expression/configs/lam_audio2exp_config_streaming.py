weight = 'pretrained_models/lam_audio2exp_streaming.tar'  # path to model weight
ex_vol = True # extract
audio_input = './assets/sample_audio/BarackObama.wav'
save_json_path = 'bsData.json'

audio_sr = 16000
fps = 30.0

movement_smooth = False
brow_movement = False
id_idx = 0

resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/audio2exp"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 100  # sche total eval & checkpoint epoch

sync_bn = False
enable_amp = False
empty_cache = False
find_unused_parameters = False

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# model settings
model = dict(
    type="DefaultEstimator",
    backbone=dict(
        type="Audio2Expression",
        pretrained_encoder_type='wav2vec',
        pretrained_encoder_path='facebook/wav2vec2-base-960h',
        wav2vec2_config_path = 'configs/wav2vec2_config.json',
        num_identity_classes=12,
        identity_feat_dim=64,
        hidden_dim=512,
        expression_dim=52,
        norm_type='ln',
        use_transformer=False,
        num_attention_heads=8,
        num_transformer_layers=6,
    ),
    criteria=[dict(type="L1Loss", loss_weight=1.0, ignore_index=-1)],
)

dataset_type = 'audio2exp'
data_root = './'
data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        test_mode=True
        ),
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="DefaultTrainer")

# Tester
infer = dict(type="Audio2ExpressionInfer",
             verbose=True)
