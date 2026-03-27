import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


# 1. 아까 저장된 CSV 파일 불러오기 (파일명 맞춰서 수정)
csv_file_path = "math_poc_results.csv"
df = pd.read_csv(csv_file_path)

print("임베딩을 추출하는 중입니다...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 문제(Q)와 초안(Draft)을 각각 독립적으로 임베딩
Q_embeddings = embedder.encode(df['problem'].tolist())      # (200, 384)
Draft_embeddings = embedder.encode(df['draft'].tolist())    # (200, 384)
y = df['label'].values

# 3. Track별 데이터 구성
# 대조군: 원본 쿼리만
X_control = Q_embeddings 

# Track A: Draft만 
X_track_A = Draft_embeddings

# Track B: Late Fusion (벡터 이어붙이기 -> 768차원)
X_track_B = np.hstack((Q_embeddings, Draft_embeddings))

# Track C: Vector Difference (벡터 빼기 -> 384차원)
X_track_C = Q_embeddings - Draft_embeddings

# 4. 로지스틱 회귀 5-Fold 교차 검증 (엄밀한 객관성 확보)
clf = LogisticRegression(max_iter=1000)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
score_control = cross_val_score(clf, X_control, y, cv=skf, scoring='accuracy').mean()
score_A = cross_val_score(clf, X_track_A, y, cv=skf, scoring='accuracy').mean()
score_B = cross_val_score(clf, X_track_B, y, cv=skf, scoring='accuracy').mean()
score_C = cross_val_score(clf, X_track_C, y, cv=skf, scoring='accuracy').mean()


print("\n" + "="*60)
print(f"📊 [대조군] 원본 문제(Q) 단독: {score_control*100:.2f}%")
print("-" * 60)
print(f"📊 [Track A] Draft 단독: {score_A*100:.2f}%")
print(f"📊 [Track B] Late Fusion (Q + Draft 결합): {score_B*100:.2f}%")
print(f"📊 [Track C] Vector Difference (Q - Draft): {score_C*100:.2f}%")
print("="*60)