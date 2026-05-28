from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

# ---------- 1. 连接 ----------
connections.connect(host="172.188.88.5", port="19530")
print("✅ 连接成功")

# ---------- 2. 创建 Collection（相当于知识库） ----------
collection_name = "test_hello_milvus"

# 如果已存在就先删掉，方便反复测试
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# 定义字段：id (主键) + text (文字) + embedding (向量)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=8),  # 向量维度
]
schema = CollectionSchema(fields, description="测试知识库")
collection = Collection(name=collection_name, schema=schema)
print(f"✅ Collection '{collection_name}' 创建成功")

# ---------- 3. 插入几条文字（用随机向量模拟 embedding） ----------
import random

texts = [
    "Milvus 是一个开源向量数据库",
    "它专门为相似性搜索而设计",
    "支持毫秒级海量向量检索",
]
# 模拟 embedding：每条文字对应一个 8 维向量
embeddings = [[random.random() for _ in range(8)] for _ in texts]

entities = [
    texts,       # text 字段
    embeddings,  # embedding 字段
]
insert_result = collection.insert(entities)
print(f"✅ 插入 {len(insert_result.primary_keys)} 条数据, IDs: {insert_result.primary_keys}")

# ---------- 4. 为向量字段创建索引 ----------
index_params = {
    "metric_type": "L2",        # 欧氏距离
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1},     # 数据量少时设小一点
}
collection.create_index(field_name="embedding", index_params=index_params)
print("✅ 索引创建成功")

# ---------- 5. 加载 Collection 到内存（搜索前必须） ----------
collection.load()

# ---------- 6. 执行向量相似性搜索 ----------
# 待搜索的向量（随机生成，实际场景由模型生成）
query_vectors = [[random.random() for _ in range(8)]]

search_params = {"metric_type": "L2", "params": {"nprobe": 1}}
results = collection.search(
    data=query_vectors,
    anns_field="embedding",
    param=search_params,
    limit=2,                      # 返回 top-2 相似结果
    output_fields=["text"],       # 同时返回原始文字
)

# 输出搜索结果
for idx, hit in enumerate(results[0]):
    print(f"  结果 {idx+1}: id={hit.id}, distance={hit.distance}, text='{hit.entity.get('text')}'")

print("✅ 搜索完成")

# 清理资源（可选）
collection.release()
connections.disconnect("default")
print("🎉 测试全部通过，Milvus 工作正常！")