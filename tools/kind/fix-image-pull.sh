#!/bin/bash
# 快速修复 Kind 集群镜像拉取问题
# 适用于中国大陆网络环境

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== 修复 Kind 集群镜像拉取问题 ===${NC}"
echo ""

cd /home/jared/vllm-project/semantic-router

echo -e "${YELLOW}步骤 1/7: 检查当前环境...${NC}"
if [[ "$CONDA_DEFAULT_ENV" != "vllm" ]]; then
    echo -e "${RED}请先激活 vllm 环境: conda activate vllm${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 环境正确${NC}"
echo ""

echo -e "${YELLOW}步骤 2/7: 构建 semantic-router 镜像（使用本地 VPN）...${NC}"
docker build -t semantic-router-extproc:local -f Dockerfile.extproc .
echo -e "${GREEN}✓ 镜像构建完成${NC}"
echo ""

echo -e "${YELLOW}步骤 3/7: 拉取 Python 基础镜像...${NC}"
docker pull python:3.11-slim
echo -e "${GREEN}✓ Python 镜像拉取完成${NC}"
echo ""

echo -e "${YELLOW}步骤 4/7: 加载 semantic-router 镜像到 Kind...${NC}"
kind load docker-image semantic-router-extproc:local --name semantic-router-cluster
echo -e "${GREEN}✓ semantic-router 镜像已加载${NC}"
echo ""

echo -e "${YELLOW}步骤 5/7: 加载 Python 镜像到 Kind...${NC}"
kind load docker-image python:3.11-slim --name semantic-router-cluster
echo -e "${GREEN}✓ Python 镜像已加载${NC}"
echo ""

echo -e "${YELLOW}步骤 6/7: 删除旧的部署...${NC}"
kubectl delete deployment semantic-router -n vllm-semantic-router-system --ignore-not-found=true
echo -e "${GREEN}✓ 旧部署已删除${NC}"
echo ""

echo -e "${YELLOW}步骤 7/7: 重新部署应用...${NC}"
kubectl apply -k deploy/kubernetes/
echo -e "${GREEN}✓ 应用已部署${NC}"
echo ""

echo -e "${GREEN}=== 部署完成！===${NC}"
echo ""
echo -e "${YELLOW}监控部署状态：${NC}"
echo -e "  kubectl get pods -n vllm-semantic-router-system -w"
echo ""
echo -e "${YELLOW}查看日志：${NC}"
echo -e "  kubectl logs -f deployment/semantic-router -n vllm-semantic-router-system"
echo ""
echo -e "${YELLOW}查看详细信息：${NC}"
echo -e "  kubectl describe pod -n vllm-semantic-router-system -l app=semantic-router"
echo ""

echo -e "${GREEN}正在监控 Pod 状态...（按 Ctrl+C 退出）${NC}"
kubectl get pods -n vllm-semantic-router-system -w
