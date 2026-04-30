#!/bin/bash

# 设置颜色输出
RED='\033[0:31m'
GREEN='\033[0:32m'
NC='\033[0m' # 无颜色

echo -e "${RED}⚠️  警告: 这将彻底删除 Milvus 所有数据和元数据 (Etcd, MinIO, Segcore)...${NC}"
read -p "确认继续执行吗? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "操作已取消。"
    exit 1
fi

# 1. 停止并移除容器及匿名卷
echo -e "${GREEN}步骤 1: 停止并清理 Docker 容器...${NC}"
docker-compose down --volumes --remove-orphans

# 2. 物理删除挂载目录 (解决 Server ID Mismatch 的核心)
echo -e "${GREEN}步骤 2: 物理删除磁盘残留数据 (sudo)...${NC}"
# 这里的目录必须对应你 yaml 中的 volumes 路径
sudo rm -rf ./volumes/etcd
sudo rm -rf ./volumes/minio
sudo rm -rf ./volumes/milvus

# 3. 清理 Docker 网络 (防止 IP 冲突导致的时钟偏移感官错误)
echo -e "${GREEN}步骤 3: 清理 Docker 网络缓存...${NC}"
docker network prune -f

# 4. 重新创建空目录 (可选，通常 docker 会自动创建，但手动创建更稳)
mkdir -p ./volumes/etcd ./volumes/minio ./volumes/milvus

# 5. 启动 Milvus
echo -e "${GREEN}步骤 4: 重新启动 Milvus 服务...${NC}"
docker-compose up -d

# 6. 等待并检查状态
echo -e "${GREEN}步骤 5: 正在等待服务就绪...${NC}"
sleep 10
docker-compose ps

echo -e "\n${GREEN}✅ 重置完成！${NC}"
echo "请运行以下命令观察日志，直到看到 'Milvus server is ready to serve!' 后再运行 Python 脚本："
echo -e "${RED}docker logs -f milvus-standalone${NC}"