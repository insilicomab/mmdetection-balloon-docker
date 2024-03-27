_base_ = ["../../mmdetection/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py"]

# configurations
work_dir = "./outputs/mask2former"
data_root = "./data/"
img_dir = {
    "train": "balloon/train",
    "val": "balloon/val",
    "test": "balloon/val",
}
annotation_path = {
    "train": "coco/train_coco_annotation.json",
    "val": "coco/val_coco_annotation.json",
    "test": "coco/val_coco_annotation.json",
}
logger_params = {
    "project": "mmdetection-balloon",
    "run_name": "mask2former_r50_8xb2-lsj-50e_coco",
}

## models
max_iters = 368750
num_things_classes = 1  # オブジェクトのクラス数
num_stuff_classes = 0  # panopticで使うstuffのクラス数, 0にすればinstance segmentation
num_classes = num_things_classes + num_stuff_classes
image_size = (1024, 1024)
classes = ("balloon",)

optimizer = "AdamW"
lr = 0.0001
weight_decay = 0.05
eps = 1e-8

## datasets
dataset_type = "CocoDataset"
batch_size = 2
metainfo = dict(classes=classes)

## default_runtime
checkpoint_params = {
    "by_epoch": False,
    "save_best": "coco/segm_mAP",
    "rule": "greater",
}
early_stopping_params = {
    "monitor": "coco/segm_mAP",
    "rule": "greater",
    "patience": 10,
    "strict": False,
}

# models
# ../../mmdetection/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py
batch_augments = [
    dict(
        type="BatchFixedSizePad",
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
    )
]
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments,
)
model = dict(
    data_preprocessor=data_preprocessor,
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1]),
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes, num_stuff_classes=num_stuff_classes
    ),
    test_cfg=dict(panoptic_on=False),
)

# ../../mmdetection/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py
## optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type=optimizer, lr=lr, weight_decay=weight_decay, eps=eps, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1, decay_mult=1.0),
            "query_embed": embed_multi,
            "query_feat": embed_multi,
            "level_embed": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=0.01, norm_type=2),
)

## learning policy
param_scheduler = dict(
    type="MultiStepLR",
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[327778, 355092],
    gamma=0.1,
)

interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# datasets
# ../../mmdetection/configs/_base_/datasets/coco_panoptic.py
train_pipeline = [
    dict(
        type="LoadImageFromFile", to_float32=True, backend_args={{_base_.backend_args}}
    ),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    # large scale jittering
    dict(
        type="RandomResize",
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type="Resize",
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_size=image_size,
        crop_type="absolute",
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(
        type="LoadImageFromFile", to_float32=True, backend_args={{_base_.backend_args}}
    ),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataset = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file=annotation_path["train"],
    data_prefix=dict(img=img_dir["train"]),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=None,
)
val_dataset = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file=annotation_path["val"],
    data_prefix=dict(img=img_dir["val"]),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=None,
)
test_dataset = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file=annotation_path["test"],
    data_prefix=dict(img=img_dir["test"]),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=None,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=train_dataset,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=test_dataset,
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + annotation_path["val"],
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=None,
)
test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + annotation_path["test"],
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=None,
)


# default_runtime
# ../../mmdetection/configs/_base_/default_runtime.py
## hooks
## https://mmdetection.readthedocs.io/en/latest/user_guides/config.html#hook-config
## https://mmengine.readthedocs.io/en/latest/tutorials/hook.html
## https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.CheckpointHook.html#mmengine.hooks.CheckpointHook
## https://mmengine.readthedocs.io/en/latest/tutorials/hook.html#checkpointhook
## https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.EarlyStoppingHook.html#mmengine.hooks.EarlyStoppingHook
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=1,
        interval=interval,
    )
)
custom_hooks = [
    dict(
        type="CheckpointHook",
        by_epoch=checkpoint_params["by_epoch"],
        save_best=checkpoint_params["save_best"],
        rule=checkpoint_params["rule"],
    ),
    dict(
        type="EarlyStoppingHook",
        monitor=early_stopping_params["monitor"],
        rule=early_stopping_params["rule"],
        min_delta=0.005,
        strict=early_stopping_params["strict"],
        check_finite=True,
        patience=early_stopping_params["patience"],
        stopping_threshold=None,
    ),
]

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    dict(
        type="MLflowVisBackend",
        save_dir="./mlruns",
        exp_name=logger_params["project"],
        run_name=logger_params["run_name"],
        tags=None,
        params=None,
        artifact_suffix=[".json", ".log", ".py", "yaml"],
    ),
    dict(
        type="WandbVisBackend",
        save_dir="./wandb",
        init_kwargs=dict(
            project=logger_params["project"], name=logger_params["run_name"]
        ),
    ),
]

visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

load_from = "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"
