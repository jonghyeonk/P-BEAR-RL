
import warnings
warnings.filterwarnings(action='ignore')
import os
import pandas as pd
import numpy as np
import plotly.io as pio
import networkx as nx
import matplotlib.pyplot as plt
import time

pio.renderers.default = 'iframe_connected'

my_path = os.path.abspath('')

from PIL import Image, ImageDraw
import random
from matplotlib.colors import ListedColormap
from collections import deque

from func import discover_NBG
from func import P_BEAR_RL, QLearningAgent, P_BEAR_DRL, DQNAgent, preprocess_state,discover_pattern

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', default=None, help='log')
parser.add_argument('--num_epi', default=100, help='episodes')
parser.add_argument('--model', default='QL', help='method')


dat_name = parser.parse_args().data
num_epi = int(parser.parse_args().num_epi)
model = parser.parse_args().model


# 파일 경로 설정
input_path = "c:\\Users\\user\\Desktop\\연구\\PBAR_extension\\alignment"
anomalous_data_path = os.path.join(input_path, "encoded_anomaly", dat_name + "_1.00.csv")

output_path = "c:\\Users\\user\\Desktop\\연구\\PBAR_extension\\PBAR4py\\output"

# 데이터 로드 및 컬럼명 변경, 데이터 타입 명시

dtype_data = {'Case': 'category', 'Activity': 'category', "type_res_trace": 'category'}
data = pd.read_csv(anomalous_data_path, usecols=['Case', 'Activity', "Timestamp", "type_res_trace"], dtype=dtype_data)
data = data.rename(columns={"Case": "case:concept:name", "Activity": "concept:name",
                           "Timestamp": "time:timestamp", "type_res_trace": "label"}, copy=False)


# 'Start' 및 'End' 액티비티 추가
data['order'] = data.groupby('case:concept:name', observed=True).cumcount() + 1

start = data[data['order'] == 1].copy()
start.loc[:, 'order'] = 0  # SettingWithCopy 경고 방지
start.loc[:, 'concept:name'] = "Start"

end = data.groupby('case:concept:name', observed=True).apply(lambda x: x[x['order'] == x['order'].max()].copy()).reset_index(drop=True)
end.loc[:, 'order'] = end['order'] + 1
end.loc[:, 'concept:name'] = "End"

data2 = pd.concat([data, start, end], ignore_index=True).sort_values(by=["case:concept:name", 'order'], kind='stable').reset_index(drop=True)

# 정상 및 이상 데이터 분리
clean = data2[data2.label.isna()].reset_index(drop=True)
anomaly = data2.reset_index(drop=True)
del data2  # 더 이상 필요없는 DataFrame 삭제


# utils
actset = clean['concept:name'].unique()
filtered_actset = [item for item in actset if item not in ['Start', 'End']]

act_freq = clean['concept:name'].value_counts()
act_freq = act_freq.drop(index=['Start','End'], errors='ignore')
act_freq = act_freq.reset_index()
act_freq['prob'] = act_freq['count']/sum(act_freq['count'])
act_freq = act_freq.set_index('concept:name').reindex(filtered_actset).reset_index()


# 시작 시간 기록
start_time = time.time()


NBGs= discover_NBG(clean)


# 학습 루프
if __name__ == "__main__":
    

    # training
    print('Start training P-BEAR-RL')
    anomaly_nolabel = anomaly[['case:concept:name', 'concept:name', 'order']]    
    caseids = anomaly_nolabel['case:concept:name'].unique()
    if model =='QL':
        # jh
        agent = QLearningAgent(filtered_actset , act_freq, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.5, exploration_decay_rate=0.0001)

        repaired_df = pd.DataFrame()
        for id in caseids:
            # agent = QLearningAgent(filtered_actset , act_freq, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.5, exploration_decay_rate=0.0001)

            print(f"Case ID: {id}")
            obs_case = anomaly_nolabel.loc[anomaly_nolabel['case:concept:name']  == id].reset_index(drop=True)

            env = P_BEAR_RL(obs_case, actset, filtered_actset, NBGs, alpha =0) # jh
            episodes = num_epi

            shortest_episode = float('inf')
            highest_reward_shortest_episode = -float('inf')
            best_episode_history = None
            all_episode_histories = []

            for episode in range(episodes):
                state = env.reset()
                total_reward = 0
                steps = 0
                done = False
                game_over = False

                episode_history = [] 

                while not done and not game_over:
                    action = agent.select_action(state)
                    next_state, reward, done, game_over, info, loc, label_act, rework = env.step(action, max_step=10)

                    agent.update_q_value(state, action, reward, next_state)
                    agent.decay_exploration_rate()

                    total_reward += reward
                    steps += 1
                    if not done and (label_act != None): #jh
                        episode_history.append((state.copy(), env.action_labels[action].split('_')[0], loc, label_act, rework,reward, next_state.copy()))
                    state = next_state

                if done and not game_over:
                    all_episode_histories.append((steps-1, total_reward, episode_history))
                    # env.render()

            if len(all_episode_histories)>0:
                check_clean = [h[0] for h in all_episode_histories]
                if max(check_clean)>0:
                    # 최종 결과 선택

                    
                    shortest_episodes = [hist for hist in all_episode_histories if hist[0] == min(h[0] for h in all_episode_histories if h[0] > 0)] # 0 step 제외

                    best_episode = max(shortest_episodes, key=lambda x: x[1])
                    env.state = best_episode[2][-1][-1] # 마지막 상태 렌더링
                    
                    env.state['predict_patterns'] = discover_pattern(best_episode[2])
                    repaired_df = pd.concat([repaired_df, env.state]).reset_index(drop= True)

                    # env.render()
                else:
                    state = env.init_state
                    state['predict_patterns'] = ''
                    repaired_df = pd.concat([repaired_df, state]).reset_index(drop= True)
            else:
                state = env.init_state
                state['predict_patterns'] = ''
                repaired_df = pd.concat([repaired_df, state]).reset_index(drop= True)


        repaired_df.to_csv(  os.path.join(output_path,dat_name +"_" + str(num_epi)+ ".csv"), index= False)

        # 종료 시간 기록
        end_time = time.time()

        # 실행 시간 계산
        execution_time = end_time - start_time

        # 실행 시간 출력 (다양한 단위로 출력 가능)
        print(f"\n총 실행 시간: {execution_time:.4f} 초")
        
        
    else:
        
        CONSISTENT_MAX_LEN = max(clean.groupby('case:concept:name').apply(len)) + 10
        actset_for_vocab = filtered_actset
        
        anomaly_nolabel = anomaly[['case:concept:name', 'concept:name', 'order']]    
        caseids = anomaly_nolabel['case:concept:name'].unique()
        
        one_hot_depth = len(filtered_actset) + 1

        agent = DQNAgent(CONSISTENT_MAX_LEN * one_hot_depth, list(range(len(filtered_actset)+2)), learning_rate=0.0005,
                        exploration_rate=0.5, exploration_decay_rate=0.0001,
                        replay_buffer_capacity=5000, batch_size=32)
        for id in caseids:

            print(f"Case ID: {id}")
            obs_case = anomaly_nolabel.loc[anomaly_nolabel['case:concept:name']  == id].reset_index(drop=True)

            env = P_BEAR_DRL(obs_case)
            episodes = 100
            
            shortest_episode = float('inf')
            highest_reward_shortest_episode = -float('inf')
            best_episode_history = None
            all_episode_histories = []

            for episode in range(episodes):
                state_raw = env.reset()
                state = preprocess_state(state_raw, CONSISTENT_MAX_LEN, actset_for_vocab)

                total_reward = 0
                steps = 0
                done = False
                game_over = False
                episode_history = [] 
                
                while not done and not game_over:
                    action = agent.select_action(state)
                    next_state_raw, reward, done, game_over, info, loc, label_act,rework = env.step(action, max_step=10)
                    next_state = preprocess_state(next_state_raw,CONSISTENT_MAX_LEN, actset_for_vocab)

                    agent.store_transition(state, action, reward, next_state, done)
                    agent.learn()
                    agent.decay_exploration_rate()
                    
                    total_reward += reward
                    steps += 1
                    
                    if not done and (label_act != None): #jh
                        episode_history.append((state_raw.copy(), env.action_labels[action].split('_')[0], loc, label_act ,rework, reward, next_state_raw.copy()))
                    
                    state_raw = next_state_raw
                    state = next_state

            
        # TESTING
        print('Start repairing...')
        repaired_df = pd.DataFrame()
        for id in caseids:
            obs_case = anomaly_nolabel.loc[anomaly_nolabel['case:concept:name']  == id].reset_index(drop=True)
            env = P_BEAR_DRL(obs_case)

            shortest_episode = float('inf')
            highest_reward_shortest_episode = -float('inf')
            best_episode_history = None
            all_episode_histories = []

            state_raw = env.reset()
            state = preprocess_state(state_raw, CONSISTENT_MAX_LEN, actset_for_vocab)

            total_reward = 0
            steps = 0
            done = False
            game_over = False
            episode_history = [] 
                
            while not done and not game_over:
                action = agent.select_action(state)
                next_state_raw, reward, done, game_over, info, loc, label_act, rework= env.step(action, max_step=10)
                next_state = preprocess_state(next_state_raw,CONSISTENT_MAX_LEN, actset_for_vocab)

                agent.store_transition(state, action, reward, next_state, done)
                agent.learn()
                agent.decay_exploration_rate()
                
                total_reward += reward
                steps += 1
                
                episode_history.append((state_raw.copy(), env.action_labels[action].split('_')[0], loc, label_act,rework ,reward, next_state_raw.copy()))
                state_raw = next_state_raw
                state = next_state
                
                
            if done and not game_over:
                print(f"Case ID: {id}, Finished after {steps-1} steps with reward {total_reward:.4f}")
                
            if game_over:
                print('game over')
                
            # summary results
            next_state['predict_patterns'] = discover_pattern(episode_history)
            repaired_df = pd.concat([repaired_df, next_state]).reset_index(drop= True)
            
            
        # 종료 시간 기록
        end_time = time.time()

        # 실행 시간 계산
        execution_time = end_time - start_time

        # 실행 시간 출력 (다양한 단위로 출력 가능)
        print(f"\n총 실행 시간: {execution_time:.4f} 초")
        repaired_df.to_csv(  os.path.join(output_path,dat_name +"_" + str(num_epi)+ ".csv"), index= False)
