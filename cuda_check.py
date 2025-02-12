#!/usr/bin/env python
import os
import torch
import torch.distributed as dist

def check_cuda():
    print("==== CUDA Info ====")
    if torch.cuda.is_available():
        print("CUDA is available!")
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")
        for i in range(num_devices):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

def check_nccl_build():
    print("\n==== NCCL Build Check ====")
    # 1. 試しに torch.cuda.nccl.version() を呼び出す
    try:
        nccl_version = torch.cuda.nccl.version()
        print(f"torch.cuda.nccl.version() returns: {nccl_version}")
    except Exception as e:
        print("Failed to get NCCL version via torch.cuda.nccl.version():", e)

    # 2. torch.distributed から NCCL のサポート状況を確認する
    if hasattr(dist, "is_nccl_available"):
        print("torch.distributed.is_nccl_available() returns:", dist.is_nccl_available())
    else:
        print("torch.distributed.is_nccl_available() is not available in this PyTorch version.")

def test_nccl_process_group():
    print("\n==== NCCL Process Group Test ====")
    # 分散処理の初期化に必要な環境変数を設定
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    try:
        # シングルプロセスでも初期化可能かテスト（rank=0, world_size=1）
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        print("NCCL process group initialization successful!")
        dist.destroy_process_group()
    except Exception as e:
        print("Failed to initialize NCCL process group:", e)

if __name__ == '__main__':
    check_cuda()
    check_nccl_build()
    test_nccl_process_group()
