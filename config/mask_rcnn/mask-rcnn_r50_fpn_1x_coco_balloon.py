# override config file
_base_ = [
    "../../mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",
]

# configurations
work_dir = "./outputs/mask_rcnn"
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
    "run_name": "mask-rcnn_r50_fpn_1x_coco",
}

## models
loss_fn = "FocalLoss"
classes = ("balloon",)

## datasets
dataset_type = "CocoDataset"
batch_size = 8
metainfo = dict(classes=classes)

## schedules
max_epochs = 20
optimizer = "SGD"
lr = 0.02
momentum = 0.9
weight_decay = 0.0001

## default_runtime
checkpoint_params = {
    "by_epoch": True,
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
# ../../mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py
## https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html
loss_cls = dict(type=loss_fn, use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes), loss_cls=loss_cls),
        mask_head=dict(num_classes=len(classes)),
    )
)


# datasets
# ../../mmdetection/configs/_base_/datasets/coco_instance.py
## https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html
## https://mmdetection.readthedocs.io/en/dev-3.x/advanced_guides/customize_dataset.html#customize-datasets
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
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


# schedules
## training schedule for 1x
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

## learning rate
## https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

## optimizer
## https://mmdetection.readthedocs.io/en/latest/user_guides/config.html#hook-config
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type=optimizer, lr=lr, momentum=momentum, weight_decay=weight_decay),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


# default_runtime
# ../../mmdetection/configs/_base_/default_runtime.py
## hooks
## https://mmdetection.readthedocs.io/en/latest/user_guides/config.html#hook-config
## https://mmengine.readthedocs.io/en/latest/tutorials/hook.html
## https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.CheckpointHook.html#mmengine.hooks.CheckpointHook
## https://mmengine.readthedocs.io/en/latest/tutorials/hook.html#checkpointhook
## https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.EarlyStoppingHook.html#mmengine.hooks.EarlyStoppingHook
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=-1),
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

load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
