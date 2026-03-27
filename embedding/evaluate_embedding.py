import os
import time
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ==========================================
# 1. API 셋팅 및 프롬프트 설정 (Gemini 2.0 Flash)
# ==========================================
os.environ["GEMINI_API_KEY"] = "AIzaSyAs9WFnqXVMVQrA_MANMNG_bva_u0kKdlk" 
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel('gemini-2.0-flash')

model_config = genai.types.GenerationConfig(
    temperature=0.0,
    top_p=0.95,
    max_output_tokens=200, 
)

SYSTEM_PROMPT = """System Prompt:
        You are an expert mathematical strategist. Your task is to provide a high-level logical plan to solve the given math problem.

        CRITICAL RULES:

        Do NOT solve the problem. Do not perform arithmetic calculations.

        Provide EXACTLY three sequential steps outlining the core logical approach.

        Keep each step extremely concise (under 15 words). Focus only on the mathematical theorems, properties, or operations required.

        NO conversational filler. NO introductory or concluding remarks. Output strictly in the format below.

        FORMAT:
        Step 1: [Define the initial setup or equation]
        Step 2: [Identify the key transformation or theorem to apply]
        Step 3: [Specify the final operation to isolate the answer]

        User Input:
        Problem: {query}"""

def get_draft_from_gemini(question: str) -> str:
    try:
        # 유저님이 설정한 {query} 자리에 실제 문제를 쏙 집어넣습니다.
        full_prompt = SYSTEM_PROMPT.format(query=question)
        response = model.generate_content(full_prompt,
                                            generation_config=model_config)       
        return response.text
    except Exception as e:
        print(f"API 오류: {e}")
        return "Error occurred."

# ==========================================
# 2. 파이썬 코드로 실제 MATH 벤치마크 불러오기 (HuggingFace)
# ==========================================
print("\n[HuggingFace] MATH 데이터셋을 다운로드 중입니다 (첫 실행 시 몇 분 걸릴 수 있음)...")

try:
    # 'hendrycks/competition_math' 대신 더 안정적인 'lighteval/MATH' 경로를 시도합니다.
    # 세부 설정인 'all'을 추가해 모든 문제를 가져옵니다.
    dataset = load_dataset('lighteval/MATH', 'all', split='train')
except Exception as e:
    print(f"\n[데이터셋 로드 오류]: {e}")
    print("HuggingFace 주소가 변경되었거나 접근이 제한되었을 수 있습니다.")
    print("'hendrycks/competition_math'로 다시 시도하거나 로컬 데이터를 준비해 주세요.")
    # 대체 경로 시도
    dataset = load_dataset('qwedsacf/competition_math', split='train')

df_math = dataset.to_pandas()

# 레벨별로 100문제씩 총 200문제 추출 
# (비용/시간 여유가 되시면 500개씩 추출하시면 논문급 데이터가 됩니다!)
df_level2 = df_math[df_math['level'] == 'Level 2'].head(100) 
df_level4 = df_math[df_math['level'] == 'Level 4'].head(100)

df = pd.concat([df_level2, df_level4]).sample(frac=1, random_state=42).reset_index(drop=True)
df['label'] = df['level'].apply(lambda x: 0 if x == 'Level 2' else 1)
print(f"데이터 준비 완료! (Level 2: 100개, Level 4: 100개, 총 200개)")

# ==========================================
# 3. 초안(Draft) 생성 및 임베딩
# ==========================================
print("\nGemini 2.0 Flash를 이용해 200문항 초안을 생성합니다... (API Rate Limit 때문에 시간 소요)")
drafts = []
for idx, row in df.iterrows():
    if idx % 20 == 0: print(f"API 호출 진행 중: {idx}/200")
    drafts.append(get_draft_from_gemini(row['problem']))
    time.sleep(2)

df['draft'] = drafts

print("\n임베딩 모델(all-MiniLM)을 통해 벡터 변환을 시작합니다...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

X1_texts = df['problem'].tolist()
X1_embeddings = embedder.encode(X1_texts) # (200, 384)

X2_texts = [f"Question: {row['problem']}\nDraft: {row['draft']}" for _, row in df.iterrows()]
X2_embeddings = embedder.encode(X2_texts) # (200, 384)

# ==========================================
# 4. PyTorch 단순 Linear 분류기 (Train / Test 완벽 분리!)
# ==========================================
print("\n[ PyTorch Linear 분류기 훈련 및 *일반화 성능(Test)* 비교 ]")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y_numpy = df['label'].values



def train_and_eval_linear_with_split(X_numpy_array, y_numpy_array, epochs=150):
    X_train, X_test, y_train, y_test = train_test_split(
        X_numpy_array, y_numpy_array, test_size=0.5, random_state=42, stratify=y_numpy_array
    )
    
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    model = nn.Sequential(nn.Linear(384, 1), nn.Sigmoid()).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 학습 진행
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        
    # 평가 (정확도 계산용)
    model.eval()
    with torch.no_grad():
        test_predictions = (model(X_te) >= 0.5).float()
        test_accuracy = (test_predictions == y_te).float().mean().item()
        
        # ⭐ 핵심 추가: 학습된 모델로 전체 200개 데이터에 대한 예측 라벨 쫙 뽑기
        X_all = torch.tensor(X_numpy_array, dtype=torch.float32).to(device)
        all_predictions = (model(X_all) >= 0.5).int().cpu().numpy().flatten()
        
    return test_accuracy, all_predictions

# 함수 호출 시 정확도와 전체 예측 배열을 모두 받음
acc1, pred1_all = train_and_eval_linear_with_split(X1_embeddings, y_numpy)
acc2, pred2_all = train_and_eval_linear_with_split(X2_embeddings, y_numpy)

# 데이터프레임에 예측 결과 컬럼으로 추가
df['pred_Q_only'] = pred1_all
df['pred_Q_Draft'] = pred2_all

print("="*60)
print(f"📊 [대조군] 단순 문제(Q) Linear 'Test 일반화' 정확도: {acc1*100:.2f}%")
print(f"📊 [실험군] 문제+초안(Q+Draft) Linear 'Test 일반화' 정확도: {acc2*100:.2f}%")
print("="*60)

# 이대로 저장하면 CSV에 예측값 2개가 예쁘게 박힘
df.to_csv("math_poc_results.csv", index=False, encoding='utf-8-sig')

# 결과 도출
acc1 = train_and_eval_linear_with_split(X1_embeddings, y_numpy)
acc2 = train_and_eval_linear_with_split(X2_embeddings, y_numpy)

print("="*60)
print(f"📊 [대조군] 단순 문제(Q) Linear 'Test 일반화' 정확도: {acc1*100:.2f}%")
print(f"📊 [실험군] 문제+초안(Q+Draft) Linear 'Test 일반화' 정확도: {acc2*100:.2f}%")
print("="*60)

csv_save_path = "math_poc_results.csv"
df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
print(f"\n최종 결과가 '{csv_save_path}'에 저장되었습니다.")
