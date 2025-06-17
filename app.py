import streamlit as st
import pandas as pd
import numpy as np
import ast
from pipeline import create_and_search_index, paper_info, paper_pool, user_update
import faiss

@st.cache_resource
def load_data():
     
     paper_df = pd.read_csv("data/papers_final.csv")
     user_df = pd.read_csv("data/user_paper_matrix.csv")
     paper_df['embedding'] = paper_df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
     paper_df = paper_df.drop(columns=['Unnamed: 0'])
     user_df['user profile'] = user_df['user profile'].apply(lambda x: np.array(ast.literal_eval(x)))
     
     return paper_df, user_df

@st.cache_resource
def load_model():
     
     from sentence_transformers import SentenceTransformer
     model = SentenceTransformer("all-MiniLM-L6-v2")
     dim = model.get_sentence_embedding_dimension()
     
     return model, dim

def init_session_state():
     
     for key, default in {
          'current_page': 'welcome',
          'logged_in': False,
          'user_id': None,
          'topic_selected': None,
          'user_profile': None,
          'selected_paper': None,
          'recommendations': None,
          'history': [],
          'history_i': [],
          'recommend_count': 0
     }.items():
          if key not in st.session_state:
               st.session_state[key] = default

def restart_session():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

def sidebars():
     with st.sidebar:
          st.markdown("### 세션 다시 시작")
          if st.button("종료"):
               restart_session()
                 
def display_selected_paper(paper_id, paper_df):
     
    st.subheader(paper_df.loc[paper_id]['title'])
    st.write(paper_df.loc[paper_id]['abstract'])

def recommend_and_display(dim, paper_df, user_df):
     
    user_id = st.session_state.user_id
    profile = st.session_state.user_profile
    count = st.session_state.recommend_count

    pool = paper_pool(user_id, user_df, paper_df) # 이미 선택된 논문은 pool에서 제외

    if count < 3:
        _, I_list = create_and_search_index(dim, pool, profile, 5)
    else:
        # shift-based recommendation
        _, knn_ids = create_and_search_index(dim, pool, profile, 3)
        oldest = st.session_state.history[0]
        latest = st.session_state.history[-1]
        shift = latest - oldest
        future = latest + shift
        _, shift_ids = create_and_search_index(dim, pool, future, 2)
        I_list = knn_ids + shift_ids

    st.session_state.recommendations = I_list

    st.divider()

    st.markdown("### 📑 추천 논문")
    for i, pid in enumerate(I_list):
        with st.container(border=True):
            st.markdown(f"**{i+1}. {paper_df.loc[pid]['title']}**")
            st.caption(paper_df.loc[pid]['year'])
            st.caption(paper_df.loc[pid]['abstract'][:300] + '...')
            if st.button(f"📌 이 논문 선택", key=f"select_{i}"):
                st.session_state.selected_paper = pid
                emb = paper_df.loc[pid]['embedding']
                st.session_state.history.append(emb)
                st.session_state.history_i.append(pid)
                user_df.loc[user_id-1, paper_df.loc[pid]['paperId']] = 1
                st.session_state.user_profile = np.mean(st.session_state.history, axis=0)
                st.session_state.recommend_count += 1
                st.rerun()
                
def welcome_page():
     
     with st.expander('😊 About this paper recommendation system demo ✌️'):
          st.write('''This demo shows our paper recommendation system implementation. 
                    Our paper recsys performs personalized recommendation, building user profiles based on embeddings of selected papers by user.
                    Furthermore, we propose SHIFT, which predicts user's plausible future interests based on user history and recommends 
                    them, introducing proper serendipity which does not disturb user's interest but has substantial relevancy to user's original interest.
                    ''')
          st.image('img/logo.png', width=250)
     if st.button('로그인'):
          st.session_state.current_page = 'login'
          st.rerun()

def login_page():
     
     st.header("🔐 Login")
     user_id = st.text_input("사용자 ID를 입력하세요 (1-5):")
     if st.button("Login"):
          if user_id.isdigit() and 1<=int(user_id)<=5:
               st.session_state.logged_in = True
               st.session_state.user_id = int(user_id)
               st.success("로그인 성공 👋")
               st.session_state.current_page = 'retrieve'
               st.rerun()
          else:
               st.error("로그인 실패 ❌ 올바른 ID를 입력하세요 (1-5)")
               
def retrieve_page(model, dim, paper_df, user_df):
     
     if st.session_state.topic_selected is None: # 주제가 선택되지 않았을 때
          st.header("📚 Select Topic")
          option = st.selectbox(
               '관심 분야를 선택하세요.',
               ('--관심 분야를 선택하세요--', 'object detection', 'image classifcation', 'contrastive learning',
               'language modeling', 'representation learning', 'transfer learning',
               'reinforcement learning', 'time series analysis', 'decoder',
               'question answering', 'robotics', 'autonomous driving',
               'deepfake detection', 'retrieval', 'semantic segmentation',
               'style transfer', 'decision making', 'machine learning')
          )
          if option != "--관심 분야를 선택하세요--" and st.button('검색'):
               st.session_state.topic_selected = option
               st.session_state.user_profile = model.encode(option)
               st.success(f"🔍 {option} 관련 논문 추천합니다.")
               recommend_and_display(dim, paper_df, user_df)
     else: # 주제가 선택됨
          if st.session_state.selected_paper is not None: # 논문을 선택했을 때
               st.markdown('## 📄 선택한 논문')
               display_selected_paper(st.session_state.selected_paper, paper_df)
          recommend_and_display(dim, paper_df, user_df)

               
def main():
     
     init_session_state() 
     sidebars()
     with st.spinner("데이터와 모델을 로드하는 중이에요! 잠시만 기다려주세요 :)"):
          paper_df, user_df = load_data()
          model, dim = load_model()
     
     
     st.title("🚀 Paper Recommendation Engine")
     page = st.session_state.current_page
     if page == 'welcome':
          welcome_page()
     elif page == 'login':
          login_page()
     elif page == 'retrieve':
          retrieve_page(model, dim, paper_df, user_df)

if __name__ == "__main__":
     main()
