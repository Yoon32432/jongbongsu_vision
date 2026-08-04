# 데이터 라벨링 가이드

1. AI Hub(aihub.or.kr)에서 "생활폐기물", "폐기물 이미지" 등으로 검색해
   활용 가능한 공개 데이터셋이 있는지 먼저 확인한다.
2. 부족하면 `scripts/collect_data.py`로 직접 촬영한 `data/raw/` 이미지를
   Roboflow 또는 CVAT에 업로드해 폴리곤(세그멘테이션) 라벨을 단다.
   클래스명은 `bag` 하나만 사용한다.
3. YOLO-seg 포맷(polygon txt)으로 export 후 `data/dataset.yaml`에 경로를
   맞춰 넣는다:

   ```yaml
   path: ../data
   train: images/train
   val: images/val
   names:
     0: bag
   ```
