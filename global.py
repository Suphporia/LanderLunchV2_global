# global_server.py
import argparse
import threading
import time
import random
from collections import deque, namedtuple

import numpy as np
import torch as T
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
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
    default=32,
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
app = FastAPI()

GLOBAL_BUFFER_SIZE = args.buffer_size
BATCH_SIZE         = args.batch_size
UPDATE_INTERVAL    = args.update_interval



buffer_lock   = threading.Lock()
weights_lock = threading.Lock()
global_memory = deque(maxlen=GLOBAL_BUFFER_SIZE) # 로컬 to 글로벌 저장 메모리
Transition    = namedtuple('Transition',
                           ['state','action','reward','next_state','done'])

#  로컬에서 업로드된 파라미터 저장소
LOCAL_WEIGHTS = {}  # client_id → state_dict
WEIGHT_EXPIRY_SEC = 1000      # seconds before weight expires

# Aggregation 인터벌 (초)
AGG_INTERVAL_SEC = 200.0 
last_agg_time    = time.time()

# JSON 요청을 위한 Pydantic 모델 (클라이언트 입력용)
class TransitionRequest(BaseModel):
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    
# 가중치 업로드를 위한 Pydantic 모델
class WeightsRequest(BaseModel):
    client_id: str
    state_dict: Dict[str, Any]
    
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
@app.post("/upload-transition")
async def upload_transition(transitions: List[TransitionRequest]):
    try:
        # 버퍼 락 획득
        if not buffer_lock.acquire(timeout=10):  # 10초 타임아웃
            return JSONResponse(
                status_code=503, 
                content={"status": "error", "message": "Server busy. Please try again later."}
            )
        
        try:
            # 전역 버퍼에 추가
            global global_memory
            
            for t in transitions:
                # JSON 데이터를 namedtuple로 변환
                transition = Transition(
                    state=t.state,
                    action=t.action,
                    reward=t.reward,
                    next_state=t.next_state,
                    done=t.done
                )
                global_memory.append(transition)
            
            # 버퍼가 가득 차면 가장 오래된 항목이 자동으로 제거
            
            print(f"Added {len(transitions)} transitions. Current buffer size: {len(global_memory)}")
            
            return {
                "status": "success",
                "added": len(transitions),
                "current_buffer_size": len(global_memory),
                "buffer_max_size": global_memory.maxlen
            }
        
        finally:
            # 락 해제
            buffer_lock.release()
    
    except Exception as e:
        # 예외 발생 시 락이 해제되지 않았다면 해제
        if buffer_lock.locked():
            buffer_lock.release()
        
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# 버퍼 확인용 라우트 (락 X)
@app.get("/view-buffer")
def view_buffer():
    try:
        buffer_content = [
            {
                "state": t.state,
                "action": t.action,
                "reward": t.reward,
                "next_state": t.next_state,
                "done": t.done
            } for t in global_memory
        ]
        
        return {
            "buffer_size": len(global_memory),
            "buffer_max_size": global_memory.maxlen,
            "buffer_content": buffer_content
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.delete("/clear-buffer")
def clear_buffer():
    try:
        # 버퍼 락 획득
        if not buffer_lock.acquire(timeout=5):  # 5초 타임아웃
            return JSONResponse(
                status_code=503, 
                content={"status": "error", "message": "Server busy. Please try again later."}
            )
        
        try:
            global global_buffer
            buffer_size_before = len(global_buffer)
            global_buffer.clear()  # deque 비우기
            
            return {
                "status": "success",
                "message": f"Successfully cleared buffer. Removed {buffer_size_before} items.",
                "current_buffer_size": 0,
                "buffer_max_size": global_buffer.maxlen
            }
        
        finally:
            # 락 해제
            buffer_lock.release()
    
    except Exception as e:
        # 예외 발생 시 락이 해제되지 않았다면 해제
        if buffer_lock.locked():
            buffer_lock.release()
        
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# 로컬 가중치 업로드 
@app.post("/upload-weights")
async def upload_weights(weights_data: WeightsRequest):
    try:
        # 가중치 락 획득
        if not weights_lock.acquire(timeout=5):  # 5초 타임아웃
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "Please try again later."}
            )
        
        try:
            global LOCAL_WEIGHTS
            
            client_id = weights_data.client_id
            state_dict_json = weights_data.state_dict
            
            # JSON 형태의 state_dict를 텐서로 변환
            state_dict = {}
            for key, value in state_dict_json.items():
                # 텐서로 변환해야 하는 경우
                if isinstance(value, list) or isinstance(value, dict):
                    state_dict[key] = T.tensor(value, dtype=T.float32)
                else:
                    state_dict[key] = value
            
            # LOCAL_WEIGHTS에 저장
            LOCAL_WEIGHTS[client_id] = (state_dict, time.time())
            print("state_dict ??????????????????????????",state_dict)
            print(f"Received weights from client {client_id}. Total clients: {len(LOCAL_WEIGHTS)}")
            print("→ incoming keys:", list(state_dict_json.keys()))
            print("→ model keys   :", list(global_agent.Q_eval.state_dict().keys()))
            
            return {
                "status": "success",
                "message": f"Weights received from client {client_id}",
                "total_clients": len(LOCAL_WEIGHTS)
            }
        
        finally:
            # 락 해제
            weights_lock.release()
    
    except Exception as e:
        # 예외 발생 시 락이 해제되지 않았다면 해제
        if weights_lock.locked():
            weights_lock.release()
        
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.delete("/clear-weights")
def clear_weights():
    try:
        # 가중치 락 획득
        if not weights_lock.acquire(timeout=5):
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "Server busy. Please try again later."}
            )
        
        try:
            global LOCAL_WEIGHTS
            weights_count_before = len(LOCAL_WEIGHTS)
            LOCAL_WEIGHTS.clear()
            
            return {
                "status": "success",
                "message": f"Successfully cleared weights from {weights_count_before} clients.",
                "current_clients": 0
            }
        
        finally:
            # 락 해제
            weights_lock.release()
    
    except Exception as e:
        # 예외 발생 시 락이 해제되지 않았다면 해제
        if weights_lock.locked():
            weights_lock.release()
        
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# 유효 파라미터 필터링 -> 너무 오랬동안 담겨져 있던 파라미터면 삭제 유효성을 위해서
def get_valid_local_weights():
    now = time.time()
    valid = []
    for client, (sd, ts) in list(LOCAL_WEIGHTS.items()):
        if now - ts <= WEIGHT_EXPIRY_SEC:
            valid.append(sd)
        else:
            del LOCAL_WEIGHTS[client]
    return valid

# Aggregation 함수
def aggregate_parameters():
    # θ_new = 0.5·θ_global + 0.5/K · sum_i θ_local_i (현재 예시시 형태로 만들어 놓은 거임)
    local_sds = get_valid_local_weights()
    K = len(local_sds)
    print("k>???????????????????????????", K)
    if K == 0:
        print("k = 0 return-----------------------")
        return
    alpha = 0.5
    beta  = 0.5 / K
    g_state = global_agent.Q_eval.state_dict()
    new_state = {}
    for key, g_param in g_state.items():
        combined = alpha * g_param.clone()
        for sd in local_sds:
            combined += beta * sd[key]
        new_state[key] = combined
    print(new_state, "new state-================================")
    global_agent.Q_eval.load_state_dict(new_state)
    global_agent.Q_next.load_state_dict(new_state)
    
    print(f"[AGG] merged {K} locals (α={alpha}, β={beta:.4f})")

# 배치 학습 &  Aggregation
def train_loop():
    global last_agg_time
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    
    print("train_loop start")
    while True:
        # --- Replay Buffer로부터 배치 샘플링 & 학습 ---
        with buffer_lock:
            print("train loop learning")
            if len(global_memory) >= BATCH_SIZE:
                batch = random.sample(global_memory, BATCH_SIZE)
            else:
                batch = None
                
        if batch is None: print("버퍼 부족, batch None")
        
        if batch:
            states      = T.tensor(np.vstack([e.state      for e in batch]),
                                dtype=T.float32, device=device)
            actions     = T.tensor(np.vstack([e.action     for e in batch]),
                                dtype=T.int64,   device=device)
            rewards     = T.tensor(np.vstack([e.reward     for e in batch]),
                                dtype=T.float32, device=device)
            next_states = T.tensor(np.vstack([e.next_state for e in batch]),
                                dtype=T.float32, device=device)
            dones       = T.tensor(np.vstack([e.done       for e in batch]).astype(np.uint8),
                                dtype=T.float32, device=device)

            global_agent.learn((states, actions, rewards, next_states, dones))

        # --- 주기적 파라미터 Aggregation ---
        now = time.time()
        if now - last_agg_time >= AGG_INTERVAL_SEC:
            print("parameter aggregation")
            aggregate_parameters()
            last_agg_time = now

        time.sleep(UPDATE_INTERVAL)


# 9) Aggregated params 다운로드 엔드포인트
# 9) 글로벌 모델 파라미터 다운로드 엔드포인트
@app.get("/download-params")
async def download_params():
    try:
        global global_agent
        
        # 글로벌 에이전트의 현재 상태 딕셔너리 가져오기
        state_dict = global_agent.Q_eval.state_dict()
        
        # PyTorch 텐서를 JSON 직렬화 가능한 형식으로 변환
        serializable_weights = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, T.Tensor):
                serializable_weights[key] = tensor.cpu().tolist()  # 텐서를 CPU로 이동 후 리스트로 변환
            else:
                serializable_weights[key] = tensor
        
        # 간단하게 데이터 타입과 키 출력
        print(f"다운로드 요청: 타입={type(serializable_weights)}, 키={list(serializable_weights.keys())}")
        
        return {
            "status": "success",
            "message": "글로벌 모델 파라미터를 성공적으로 가져왔습니다",
            "weights": serializable_weights
        }
    
    except Exception as e:
        print(f"오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"파라미터 가져오기 오류: {str(e)}"}
        )        
# --------------------
# 실행
# --------------------
if __name__ == '__main__':
    print("[MAIN] starting train_loop thread")
    threading.Thread(target=train_loop,  daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)



# 글로벌에서 로컬로 새롭게 업데이트 된 파라미터 주면, 교체해서 다시 학습 진행하는 과정 추가해야함.