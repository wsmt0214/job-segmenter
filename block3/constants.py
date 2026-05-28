"""Block3 상수"""

BLOCK3_TYPES = ("model", "beauty", "photo")

# 타입별 KMeans 탐색 범위 (양 끝 포함)
K_RANGE_BY_TYPE: dict[str, range] = {
    "model": range(8, 16),   # 8 .. 15
    "beauty": range(5, 11),  # 5 .. 10
    "photo": range(4, 6),    # 4 .. 5
}

CLASSIFIER_PKL = {
    "model": "classifier_model.pkl",
    "beauty": "classifier_beauty.pkl",
    "photo": "classifier_photo.pkl",
}
