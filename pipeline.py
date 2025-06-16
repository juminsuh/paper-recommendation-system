import pandas as pd
import numpy as np
import random

import faiss
import ast

import streamlit as st

from sentence_transformers import SentenceTransformer

paper_df = pd.read_csv("data/papers_final.csv")
user_df = pd.read_csv("data/user_paper_matrix.csv")

paper_df['embedding'] = paper_df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
paper_df = paper_df.drop(columns=['Unnamed: 0'])
user_df['user profile'] = user_df['user profile'].apply(lambda x: np.array(ast.literal_eval(x)))

model = SentenceTransformer("all-MiniLM-L6-v2")
dim = 384

def create_and_search_index(embedding_size, paper_df, query, knn):

    x = np.stack(paper_df['embedding'].values) # (?, 384): 사용자가 이미 읽은 논문은 pool에서 제외됨
    q = query.reshape(1, -1) # (1, 384)

    index = faiss.IndexFlatL2(embedding_size)
    index.add(x)

    k = knn
    D, I = index.search(q, k) # L2 거리, 논문의 인덱스

    D_list = D[0].tolist()
    I_list = I[0].tolist()

    return D_list, I_list # 5개씩 리스트

def paper_info(id_list, dis_list, papers):

    for i in range(len(id_list)):
        print(f"논문 번호: {i+1}")
        print(f"제목: {papers.loc[id_list[i], 'title']}")
        print(f"초록: {papers.loc[id_list[i], 'abstract']}")
        print(f"L2 거리: {dis_list[i]:.5f}")
        print('\n')
        
def paper_pool(user_id, user_df, paper_df):

    arxiv_cols_df = user_df.drop(columns=['user profile'])
    col_with_1 = arxiv_cols_df.columns[arxiv_cols_df.loc[user_id-1] == 1].tolist()
    pool = paper_df[~paper_df['paperId'].isin(col_with_1)].set_index('paperId')

    return pool # user_id 사용자가 아직 읽지 않은 논문들만 반환 (?,)

def user_update(n, user_id, id_list, paper_df, user_df, paper_history, paper_history_i):

    idx = id_list[n-1] # 사용자가 선택한 논문 인덱스
    paper_history_i.append(idx)
    paper_id = paper_df.iloc[idx]['paperId'] # 논문의 arxiv id
    embedding = paper_df.iloc[idx]['embedding'] # 논문의 임베딩 (384,)
    paper_history.append(embedding) # 선택한 논문 임베딩을 history에 저장 (384,)

    user_df.loc[user_id-1, paper_id] = 1 # 해당 논문을 읽음으로 바꿈
    user_df.at[user_id-1, 'user profile'] = np.mean(paper_history, axis=0) # user profile 업데이트 (384,)

    return user_df.loc[user_id-1, 'user profile'], paper_history, paper_history_i # 업데이트된 user profile 반환

def main():
    # ---- start----
    paper_history_i = []
    paper_history = []
    user_id = int(input("ID: "))
    query = input("search for: ")
    print()

    q_emb = model.encode(query)
    D_list, I_list = create_and_search_index(dim, paper_df, q_emb, 5)
    print(f"I_list: {I_list}") # 선택된 논문들이 중복되지 않는지 확인
    paper_info(I_list, D_list, paper_df)

    # ---- SETUP READY ----
    for i in range(3):

        n = int(input("논문 번호를 선택하세요 (1-5): ")) # 사용자가 논문 선택
        print()
        if n == 0:
            return
        user_profile, paper_history, paper_history_i = user_update(n, user_id, I_list, paper_df, user_df, paper_history, paper_history_i) # 사용자 정보 업데이트
        print(f"paper_history_i: {paper_history_i}")
        if i == 2:
            break # 3번의 user profile 업데이트가 끝났다면 이제 shift할 차례
        pool = paper_pool(user_id, user_df, paper_df)
        D_list, I_list = create_and_search_index(dim, pool, user_profile, 5)
        print(f"I_list: {I_list}")
        paper_info(I_list, D_list, paper_df)
        print("----------------------------------------------------")

    # ---- SHIFT ----

    while True:

        ############ KNN CODE ##############
        pool = paper_pool(user_id, user_df, paper_df)
        D_list, I_list = create_and_search_index(dim, pool, user_profile, 3)

        ############ SHIFT CODE #############
        oldest = paper_history[0]

        latest_paper = paper_history[-1]
        shift = latest_paper - oldest

        future = latest_paper + shift
        dist, idx = create_and_search_index(dim, pool, future, 2)

        D_list = D_list + dist
        I_list = I_list + idx
        print(f"I_list: {I_list}")

        paper_info(I_list, D_list, paper_df)

        n = int(input("논문 번호를 선택하세요 (1-5): ")) # 사용자가 논문 선택
        if n == 0:
            return
        print()
        user_profile, paper_history, paper_history_i = user_update(n, user_id, I_list, paper_df, user_df, paper_history, paper_history_i) # 사용자 정보 업데이트
        print(f"paper_history_i: {paper_history_i}")
        print("----------------------------------------------------")
        
if __name__ == "__main__":
    main()