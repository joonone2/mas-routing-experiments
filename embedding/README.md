## Query + Draft Embedding Test

### 1. 자연어 단계에서 Query와 Draft를 합쳐서 임베딩
(embedding/nl_query+embdeing.py)

**결과**

📊 [대조군] Query만 Embedding 정확도: 62.00%  
📊 [실험군] 문제+초안(Q+Draft) Embedding 정확도: 64.00%


### 2. 여러가지 vector 테스트 
(embedding/vector_test.py)  
k-fold 교차검증으로 실험(k=5)  

[대조군] 원본 문제(Q) 단독: 문제 텍스트만 보고 난이도를 맞출 때의 정확도  
[Track A] Draft 단독: 문제 없이 "해결 계획(Draft)"만 보고도 이 문제가 어려운 단계인지(Level 4) 쉬운 단계인지(Level 2) 구분이 가능한지 확인  
[Track B] Late Fusion: 문제 벡터와 초안 벡터를 횡으로 이어붙여(Concatenate, 768차원) 두 정보를 모두 온전히 활용  
[Track C] Vector Difference: 문제 벡터에서 초안 벡터를 단순 요소별 뺄셈. (문제에서 해결 계획을 뺀 나머지 추상적인 특성"이나 "변환 과정"에 난이도 정보가 있는지 확인)  


**결과**

**MiniLM-L6-v2 모델 사용**

📊 [대조군] 원본 문제(Q) 단독: 63.50%  
📊 [Track A] Draft 단독: 65.50%  
📊 [Track B] Late Fusion (Q + Draft 결합): 64.50%  
📊 [Track C] Vector Difference (Q - Draft): 61.00%  

**bge-large-en-v1.5 모델 사용**

📊 [대조군] 원본 문제(Q) 단독: 66.00%  
📊 [Track A] Draft 단독: 65.50%  
📊 [Track B] Late Fusion (Q + Draft 결합): 67.50%  
📊 [Track C] Vector Difference (Q - Draft): 60.00%  

**추이는 비슷한듯**
