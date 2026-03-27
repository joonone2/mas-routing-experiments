import os
import ssl
import urllib3

# 모든 종류의 SSL 검증 무력화 (연구/테스트용)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

csv_file_path = "math_poc_results.csv"
df = pd.read_csv(csv_file_path)

print("[SOTA 모델 로드 중...] BAAI/bge-large-en-v1.5 (최초 1회 다운로드에 1~2분 소요)")
# 기존 MiniLM 대신 무거운 SOTA 모델로 교체
embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')

print("임베딩을 추출하는 중입니다... (1024차원 변환 중)")
Q_embeddings = embedder.encode(df['problem'].tolist())
Draft_embeddings = embedder.encode(df['draft'].tolist())
y = df['label'].values

# 3. Track별 데이터 구성
X_control = Q_embeddings 
X_track_A = Draft_embeddings
X_track_B = np.hstack((Q_embeddings, Draft_embeddings))
X_track_C = Q_embeddings - Draft_embeddings

print("교차 검증(5-Fold) 진행 중...")
# 벡터 차원이 2배(768 -> 2048)로 커졌으므로 max_iter를 2000으로 늘려줌
clf = LogisticRegression(max_iter=2000)

score_control = cross_val_score(clf, X_control, y, cv=5, scoring='accuracy').mean()
score_A = cross_val_score(clf, X_track_A, y, cv=5, scoring='accuracy').mean()
score_B = cross_val_score(clf, X_track_B, y, cv=5, scoring='accuracy').mean()
score_C = cross_val_score(clf, X_track_C, y, cv=5, scoring='accuracy').mean()

print("\n" + "="*60)
print(f"🚀 [SOTA 모델 적용 결과: BAAI/bge-large-en-v1.5]")
print("-" * 60)
print(f"📊 [대조군] 원본 문제(Q) 단독: {score_control*100:.2f}%")
print(f"📊 [Track A] Draft 단독: {score_A*100:.2f}%")
print(f"📊 [Track B] Late Fusion (Q + Draft 결합): {score_B*100:.2f}%")
print(f"📊 [Track C] Vector Difference (Q - Draft): {score_C*100:.2f}%")
print("="*60)