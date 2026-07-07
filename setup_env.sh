#!/bin/bash
# ElasticPrune 环境搭建（8x3090 机器上运行）
# 国内机器: bash setup_env.sh cn   （启用清华 pip 源 + HF 镜像）
set -e

if [ "$1" = "cn" ]; then
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  export HF_ENDPOINT=https://hf-mirror.com
  # 写入 shell 配置，保证之后所有下载走镜像
  grep -q HF_ENDPOINT ~/.bashrc || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
fi

conda create -n elastic python=3.10 -y
source activate elastic

# 3090 = Ampere, CUDA 12.1 wheel 即可
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

pip install "transformers>=4.44,<4.50" accelerate sentencepiece protobuf
pip install datasets pillow numpy pandas matplotlib seaborn

# 评测框架
pip install lmms-eval

# flash-attn 可选（3090 支持，编译慢；失败不影响 Phase 1）
pip install flash-attn --no-build-isolation || echo "flash-attn 安装失败，先跳过"

# baseline 官方仓库（复现对照用）
mkdir -p baselines && cd baselines
git clone https://github.com/pkunlp-icler/FastV.git
git clone https://github.com/Gumpest/SparseVLMs.git
git clone https://github.com/dvlab-research/VisionZip.git
git clone https://github.com/Theia-4869/CDPruner.git

echo "完成。先跑: python -m elasticprune.smoke_test"
