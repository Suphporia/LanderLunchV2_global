# global_server.py
import argparse
import threading
import time
import random
from collections import deque, namedtuple

import numpy as np
import torch as T
from flask import Flask, request, jsonify
import gym

from agent import Agent
from constants import SIMPLE, DOUBLE, DUELING

# --------------------
# 1) 커맨드라인 인자 파싱
# --------------------
parser = argparse.ArgumentParser(
    description="Global DQN Learner Server for LunarLander-v2",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '-a','--agent_mode',
    choices=[SIMPLE, DOUBLE],
    default=SIMPLE,
    help='Agent Type: SIMPLE (vanilla DQN) or DOUBLE (Double DQN)'
)
parser.add_argument(
    '-n','--network_mode',
    choices=[SIMPLE, DUELING],
    default=SIMPLE,
    help='Network Type: SIMPLE or DUELING'
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='Mini-batch size for learning'
)
parser.add_argument(
    '--update_interval',
    type=float,
    default=0.5,
    help='Seconds between each global learning step'
)
parser.add_argument(
    '--buffer_size',
    type=int,
    default=1_000_000,
    help='Maximum size of the global replay buffer'
)
parser.add_argument(
    '--port',
    type=int,
    default=5050,
    help='Port for Flask server'
)
args = parser.parse_args()

# --------------------
# 2) Flask 앱 및 글로벌 버퍼 설정
# --------------------
app = Flask(__name__)

GLOBAL_BUFFER_SIZE = args.buffer_size
BATCH_SIZE         = args.batch_size
UPDATE_INTERVAL    = args.update_interval



buffer_lock   = threading.Lock()
global_memory = deque(maxlen=GLOBAL_BUFFER_SIZE) # 로컬 to 글로벌 저장 메모리
Transition    = namedtuple('Transition',
                           ['state','action','reward','next_state','done'])

#  로컬에서 업로드된 파라미터 저장소
LOCAL_WEIGHTS = {}  # client_id → state_dict

# Aggregation 설정
AGG_EPISODE_INTERVAL       = 300
TD_IMPROVEMENT_THRESHOLD   = 0.05
episode_count              = 0
last_mean_td               = None

# --------------------
# 3) 글로벌 에이전트 초기화
# --------------------
env = gym.make("LunarLander-v2")
global_agent = Agent(
    input_dims=env.observation_space.shape,
    n_actions=env.action_space.n,
    seed=0,
    agent_mode=args.agent_mode,
    network_mode=args.network_mode,
    test_mode=False,
    batch_size=BATCH_SIZE,
    update_every=1
)
# 단일 네트워크로 운용할 경우
global_agent.Q_next = global_agent.Q_eval

# --------------------
# 4) 로컬 전이 수신 엔드포인트
# --------------------
@app.route('/upload-transition', methods=['POST'])
def upload_transition():
    data = request.get_json()  # List[dict]
    with buffer_lock:
        for t in data:
            global_memory.append(Transition(
                np.array(t['state'],      dtype=np.float32),
                int(t['action']),
                float(t['reward']),
                np.array(t['next_state'], dtype=np.float32),
                bool(t['done'])
            ))
    return jsonify({'status': 'ok', 'size': len(global_memory)}), 200

# 로컬 가중치 업로드 
@app.route('/upload-weights', methods=['POST'])
def upload_weights():
    # 로컬 에이전트가 에피소드 AGG_EPISODE_INTERVAL마다 자신의 state_dict를 보냄
    # JSON으로 받아 파이토치 텐서로 복원하여 LOCAL_WEIGHTS에 저장
    payload   = request.get_json()
    client_id = payload['client_id'] # 어떤 로컬
    sd_json   = payload['state_dict'] # 파라미터터
    # JSON 리스트 → tensor 변환하는 거거
    state_dict = {k: T.tensor(v, dtype=T.float32)
                  for k, v in sd_json.items()}
    LOCAL_WEIGHTS[client_id] = state_dict
    return jsonify({'status': 'ok'}), 200


# Aggregation 함수
def aggregate_parameters():
    # θ_new = 0.5·θ_global + 0.5/K · sum_i θ_local_i (현재 예시시 형태로 만들어 놓은 거임)
    global global_agent
    K = len(LOCAL_WEIGHTS)
    if K == 0:
        return
    alpha = 0.5
    beta  = 0.5 / K

    g_state    = global_agent.Q_eval.state_dict()
    new_state  = {}

    for key, g_param in g_state.items():
        combined = alpha * g_param.clone()
        for sd in LOCAL_WEIGHTS.values():
            combined += beta * sd[key]
        new_state[key] = combined

    global_agent.Q_eval.load_state_dict(new_state)
    global_agent.Q_next.load_state_dict(new_state)
    print(f"[AGG] Aggregated params with α={alpha}, β={beta:.4f}")

#  글로벌 모델 학습 파라미터와 TD error 기반 트리거
def env_runner():
    global episode_count, last_mean_td

    while True:
        state = env.reset()
        done  = False
        td_list = []

        while not done:
            action = global_agent.choose_action(state)
            next_s, r, done, _ = env.step(action)

            # TD error 계산
            s_v      = T.tensor(state, dtype=T.float32).unsqueeze(0).to(global_agent.Q_eval.device)
            ns_v     = T.tensor(next_s, dtype=T.float32).unsqueeze(0).to(global_agent.Q_eval.device)
            q_eval   = global_agent.Q_eval(s_v)[0, action].item()
            q_next   = global_agent.Q_next(ns_v).detach().max(1)[0].item()
            td_error = abs(r + global_agent.gamma * q_next * (1 - done) - q_eval)
            td_list.append(td_error)

            # 전이 저장
            with buffer_lock:
                global_memory.append(Transition(state, action, r, next_s, done))

            state = next_s

        episode_count += 1
        global_agent.epsilon_decay()

        # 주기 & TD 개선율 확인 확인해서 업데이트 할 지 말지 결정정
        if episode_count % AGG_EPISODE_INTERVAL == 0:
            mean_td = sum(td_list) / len(td_list)
            if last_mean_td is None or (last_mean_td - mean_td) / last_mean_td > TD_IMPROVEMENT_THRESHOLD:
                print(f"[AGG] Ep {episode_count}: TD {last_mean_td}→{mean_td:.4f}, aggregating")
                aggregate_parameters()
            else:
                print(f"[AGG] Ep {episode_count}: TD improvement {(last_mean_td - mean_td):.4f} too small, skip")
            last_mean_td = mean_td

# --------------------
# 8) 글로벌 배치 학습 루프
# --------------------
def train_loop():
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    while True:
        with buffer_lock:
            if len(global_memory) >= BATCH_SIZE:
                batch = random.sample(global_memory, k=BATCH_SIZE)
            else:
                batch = None

        if batch:
            states      = T.tensor(
                np.vstack([e.state      for e in batch]), device=device)
            actions     = T.tensor(
                np.vstack([e.action     for e in batch]), device=device).long()
            rewards     = T.tensor(
                np.vstack([e.reward     for e in batch]), device=device)
            next_states = T.tensor(
                np.vstack([e.next_state for e in batch]), device=device)
            dones       = T.tensor(
                np.vstack([e.done       for e in batch]).astype(np.uint8),
                device=device
            ).float()

            global_agent.learn((states, actions, rewards, next_states, dones))

        time.sleep(UPDATE_INTERVAL)


# 9) Aggregated params 다운로드 엔드포인트
@app.route('/download-params', methods=['GET'])
def download_params():
    sd = global_agent.Q_eval.state_dict()
    sd_json = {k: v.cpu().numpy().tolist() for k, v in sd.items()}
    return jsonify({'state_dict': sd_json}), 200

# --------------------
# 실행
# --------------------
if __name__ == '__main__':
    threading.Thread(target=env_runner, daemon=True).start()
    threading.Thread(target=train_loop,  daemon=True).start()
    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)



# 글로벌에서 로컬로 새롭게 업데이트 된 파라미터 주면, 교체해서 다시 학습 진행하는 과정 추가해야함.