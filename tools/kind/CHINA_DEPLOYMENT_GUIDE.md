# Kind 集群镜像拉取问题解决方案（中国大陆）

## 问题

在中国大陆使用 Kind 部署时，即使本地有 VPN，Kind 集群内的容器运行时也无法直接拉取 GitHub Container Registry (ghcr.io) 的镜像。

## 解决方案

### 方案 1：本地构建并加载镜像（推荐）⭐

这是最可靠的方法，利用你本地的 VPN 连接构建镜像，然后加载到 Kind 集群。

#### 步骤：

1. **确保在 vllm 环境中运行**：

```bash
conda activate vllm  # 如果还没激活的话
cd /home/jared/vllm-project/semantic-router
```

2. **构建镜像（使用本地 VPN）**：

```bash
docker build -t semantic-router-extproc:local -f Dockerfile.extproc .
```

3. **加载镜像到 Kind 集群**：

```bash
kind load docker-image semantic-router-extproc:local --name semantic-router-cluster
```

4. **更新 Kubernetes 配置使用本地镜像**：

编辑 `deploy/kubernetes/kustomization.yaml`，修改 images 部分：

```yaml
images:
  - name: ghcr.io/vllm-project/semantic-router/extproc
    newName: semantic-router-extproc
    newTag: local
```

5. **重新部署**：

```bash
# 删除旧的部署
kubectl delete deployment semantic-router -n vllm-semantic-router-system

# 重新应用配置
kubectl apply -k deploy/kubernetes/

# 监控部署状态
kubectl get pods -n vllm-semantic-router-system -w
```

---

### 方案 2：使用自动化脚本

我已经创建了自动化脚本来帮你完成上述步骤：

```bash
conda activate vllm
cd /home/jared/vllm-project/semantic-router
./tools/kind/build-and-load-image.sh
```

**注意**：需要修复脚本中的集群名称检测问题。如果脚本失败，请手动执行方案 1 的步骤。

---

### 方案 3：配置 Kind 节点使用代理

这个方法让 Kind 集群节点能够使用你的代理服务器。

#### 步骤：

1. **获取主机 IP**：

```bash
HOST_IP=$(hostname -I | awk '{print $1}')
echo "Host IP: $HOST_IP"
```

2. **配置 Kind 节点代理**：

对于每个节点（control-plane 和 worker），执行：

```bash
# Control plane
docker exec semantic-router-cluster-control-plane bash -c "mkdir -p /etc/systemd/system/docker.service.d"
docker exec semantic-router-cluster-control-plane bash -c "cat > /etc/systemd/system/docker.service.d/http-proxy.conf << 'EOF'
[Service]
Environment=\"HTTP_PROXY=http://${HOST_IP}:7897\"
Environment=\"HTTPS_PROXY=http://${HOST_IP}:7897\"
Environment=\"NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.svc.cluster.local\"
EOF"

# Worker
docker exec semantic-router-cluster-worker bash -c "mkdir -p /etc/systemd/system/docker.service.d"
docker exec semantic-router-cluster-worker bash -c "cat > /etc/systemd/system/docker.service.d/http-proxy.conf << 'EOF'
[Service]
Environment=\"HTTP_PROXY=http://${HOST_IP}:7897\"
Environment=\"HTTPS_PROXY=http://${HOST_IP}:7897\"
Environment=\"NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.svc.cluster.local\"
EOF"

# 重启 containerd
docker exec semantic-router-cluster-control-plane systemctl daemon-reload
docker exec semantic-router-cluster-control-plane systemctl restart containerd
docker exec semantic-router-cluster-worker systemctl daemon-reload
docker exec semantic-router-cluster-worker systemctl restart containerd
```

3. **确保代理可以从容器访问**：

需要确保你的代理（localhost:7897）可以从 Docker 容器访问。可能需要修改代理设置允许来自 Docker 网络的连接。

4. **重启部署**：

```bash
kubectl rollout restart deployment/semantic-router -n vllm-semantic-router-system
kubectl get pods -n vllm-semantic-router-system -w
```

---

### 方案 4：使用国内镜像源

如果镜像已经推送到国内的镜像仓库（如阿里云、腾讯云），可以修改配置使用这些镜像源。

编辑 `deploy/kubernetes/kustomization.yaml`：

```yaml
images:
  - name: ghcr.io/vllm-project/semantic-router/extproc
    newName: registry.cn-hangzhou.aliyuncs.com/your-namespace/semantic-router-extproc
    newTag: latest
```

---

## 推荐执行流程

**最简单可靠的方式是方案 1（本地构建）**：

```bash
# 1. 切换环境
conda activate vllm

# 2. 进入项目目录
cd /home/jared/vllm-project/semantic-router

# 3. 构建镜像（会使用你的 VPN）
docker build -t semantic-router-extproc:local -f Dockerfile.extproc .

# 4. 加载到 Kind
kind load docker-image semantic-router-extproc:local --name semantic-router-cluster

# 5. 更新配置文件
# 编辑 deploy/kubernetes/kustomization.yaml，将镜像改为：
#   newName: semantic-router-extproc
#   newTag: local

# 6. 重新部署
kubectl delete deployment semantic-router -n vllm-semantic-router-system
kubectl apply -k deploy/kubernetes/
kubectl get pods -n vllm-semantic-router-system -w
```

---

## 验证部署

```bash
# 查看 Pod 状态
kubectl get pods -n vllm-semantic-router-system

# 查看详细信息
kubectl describe pod -n vllm-semantic-router-system -l app=semantic-router

# 查看日志
kubectl logs -f deployment/semantic-router -n vllm-semantic-router-system

# 检查镜像
kubectl get pods -n vllm-semantic-router-system -o jsonpath='{.items[*].spec.containers[*].image}'
```

---

## 常见问题

### Q: init 容器仍然失败，提示无法拉取 python:3.11-slim

A: 同样需要预先拉取这个基础镜像：

```bash
# 本地拉取
docker pull python:3.11-slim

# 加载到 Kind
kind load docker-image python:3.11-slim --name semantic-router-cluster
```

### Q: Hugging Face 模型下载失败

A: 可以使用 Hugging Face 镜像站点：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后在 deployment.yaml 的 init 容器中添加环境变量：

```yaml
env:
  - name: HF_ENDPOINT
    value: "https://hf-mirror.com"
```

---

## 其他注意事项

1. **init 容器下载模型**：deployment 中的 init 容器需要从 Hugging Face 下载模型，这也可能因为网络问题失败。建议在 VPN 环境下先本地下载模型，然后挂载到容器中。

2. **资源限制**：当前配置需要较多资源（6Gi 内存）。如果你的机器资源有限，可以进一步调整 `deploy/kubernetes/deployment.yaml` 中的资源限制。

3. **持久化存储**：模型使用 PVC 存储，确保 Kind 集群的存储类可用：

```bash
kubectl get storageclass
```
