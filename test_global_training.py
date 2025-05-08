# test_global_training.py

import threading
import time
import random
import numpy as np
import torch as T
from collections import deque, namedtuple

from agent import Agent
from constants import SIMPLE
import gym

GLOBAL_BUFFER_SIZE = 100_000
BATCH_SIZE = 32
UPDATE_INTERVAL = 0.2   # 학습 루프 슬립 타임 (초)

Transition = namedtuple('Transition',
                        ['state','action','reward','next_state','done'])
global_memory = deque(maxlen=GLOBAL_BUFFER_SIZE)
buffer_lock = threading.Lock()

#더미미
def generate_dummy_transitions(num):
    """
    LunarLander 환경과 같은 모양(8차원 관측, 4개 액션)을
    랜덤하게 만들어서 global_memory에 채워넣는다.
    """
    for _ in range(num):
        s  = np.random.uniform(-1,1, size=(8,)).astype(np.float32)
        a  = random.randrange(4)
        r  = random.uniform(-1,1)
        ns = np.random.uniform(-1,1, size=(8,)).astype(np.float32)
        d  = random.random() < 0.1
        with buffer_lock:
            global_memory.append(Transition(s, a, r, ns, d))


env = gym.make("LunarLander-v2")
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

global_agent = Agent(
    input_dims=env.observation_space.shape,
    n_actions=env.action_space.n,
    seed=42,
    agent_mode=SIMPLE,
    test_mode=False,
    batch_size=BATCH_SIZE,
    update_every=1,    
    lr=1e-3
)
# 하나의 네트워크만 쓸 거라면 Q_next를 Q_eval로 덮어도 무방
global_agent.Q_next = global_agent.Q_eval


def train_loop():
    while True:
        with buffer_lock:
            if len(global_memory) < BATCH_SIZE:
                # 충분한 transition이 아니면 기다림
                pass
            else:
                # batch resampling 함함
                batch = random.sample(global_memory, k=BATCH_SIZE)
        
        # tensor 변환환
        states      = T.tensor(
            np.vstack([t.state for t in batch]), device=device)
        actions     = T.tensor(
            np.vstack([t.action for t in batch]), device=device).long()
        rewards     = T.tensor(
            np.vstack([t.reward for t in batch]), device=device)
        next_states = T.tensor(
            np.vstack([t.next_state for t in batch]), device=device)
        dones       = T.tensor(
            np.vstack([t.done for t in batch]).astype(np.uint8),
            device=device).float()

        # 한 스텝 학습
        global_agent.learn((states, actions, rewards, next_states, dones))

        #  평균 Q값
        # q_eval = global_agent.Q_eval.forward(states).gather(1, actions)
        # print("Avg Q:", q_eval.mean().item())

        time.sleep(UPDATE_INTERVAL)

# ——— 5) 테스트 실행 ———
if __name__ == '__main__':
    # 1) 더미 데이터 미리 채우기
    print("==> Generating 5000 dummy transitions...")
    generate_dummy_transitions(5000)
    print("Current buffer size:", len(global_memory))

    # 2) 학습 스레드 시작
    learner_thread = threading.Thread(target=train_loop, daemon=True)
    learner_thread.start()

    # 3) 메인 스레드: 일정 시간마다 버퍼 크기와 epsilon 변화 감시
    for i in range(10):
        time.sleep(1.0)
        with buffer_lock:
            buf_sz = len(global_memory)
        print(f"[{i}] Buffer size: {buf_sz}, Epsilon: {global_agent.epsilon:.3f}")
        global_agent.epsilon_decay()
