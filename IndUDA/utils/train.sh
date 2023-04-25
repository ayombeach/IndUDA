CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg ./experiments/config/Office-31/CAN/office31_train_amazon2webcam_cfg.yaml --method INN --exp_name office31_a2w
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg ./experiments/config/Office-31/CAN/office31_train_amazon2dslr_cfg.yaml --method INN --exp_name office31_a2d
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg ./experiments/config/Office-31/CAN/office31_train_webcam2dslr_cfg.yaml --method INN --exp_name office31_w2d
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg ./experiments/config/Office-31/CAN/office31_train_webcam2amazon_cfg.yaml --method INN --exp_name office31_w2a
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg ./experiments/config/Office-31/CAN/office31_train_dslr2webcam_cfg.yaml --method INN --exp_name office31_d2w
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg ./experiments/config/Office-31/CAN/office31_train_dslr2amazon_cfg.yaml --method INN --exp_name office31_d2a