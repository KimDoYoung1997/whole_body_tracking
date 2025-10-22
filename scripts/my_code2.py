"""
Beyond Mimic Sim2Sim MuJoCo Deploy Script

이 스크립트는 논문의 Beyond Mimic 방법론을 구현한 sim-to-sim 배포 시스템입니다.
Isaac Lab에서 학습된 정책을 MuJoCo 환경에서 실행하여 모션 트래킹을 수행합니다.

=== Sim-to-Sim Deploy 핵심 원리 ===

1. 좌표계 독립성 확보:
   - MuJoCo (Z-up) vs Isaac Lab (Y-up) 좌표계 차이에도 불구하고 작동
   - 상대적 관찰값 사용으로 절대 좌표계 차이 흡수
   - 앵커링 메커니즘으로 로봇-모션 데이터 간 상대적 정렬 유지

2. 논문의 Observation 구성 구현:
   o = [c, ξ_{b_anchor}, V_{b_root}, q_joint,r, v_joint,r, a_last]
   - c ∈ ℝ^58 : Reference Motion의 관절 위치 및 속도 (29+29)
   - ξ_{b_anchor} ∈ ℝ^9 : Anchor Body의 자세 추적 오차 (3+6)
   - V_{b_root} ∈ ℝ^6 : Robot's root twist expressed in root frame (3+3)
   - q_joint,r ∈ ℝ^29 : 로봇의 모든 Joint의 현재 각도 (상대값)
   - v_joint,r ∈ ℝ^29 : 로봇의 모든 Joint의 현재 각속도 (절대값)
   - a_last ∈ ℝ^29 : Policy가 직전에 취한 행동 (메모리 역할)

3. Policy Inference 과정:
   - ONNX 모델을 통한 실시간 추론 (50Hz)
   - 앵커링을 통한 좌표계 변환 없이 모션 트래킹
   - PD 제어기를 통한 관절 토크 계산 및 적용

=== 데이터 구조 ===
- NPZ 파일: Isaac Lab에서 export된 모션 데이터
  * body_pos_w: Isaac Lab의 30개 body 순서 (인덱스 9 = torso_link)
  * joint_pos: Reference motion의 관절 위치 (29차원)
  * joint_vel: Reference motion의 관절 속도 (29차원)
- ONNX 모델: Isaac Lab에서 export된 학습된 정책
  * 메타데이터: joint_names, default_joint_pos, action_scale 등
"""

import time
import onnx
from datetime import datetime

import mujoco.viewer
import mujoco
import numpy as np
import torch
from modules.metrics_n_plots import calculate_additional_metrics, save_performance_plots
import onnxruntime

def quat_to_rotation_matrix(quat):
    """쿼터니언을 3x3 회전 행렬로 변환"""
    rotm = np.zeros(9)
    mujoco.mju_quat2Mat(rotm, quat)
    return rotm.reshape(3, 3)

def pose_to_transformation_matrix(pos, quat):
    """위치와 쿼터니언을 4x4 transformation matrix로 변환
    
    Args:
        pos: 위치 벡터 (3,) [x, y, z]
        quat: 쿼터니언 (4,) [w, x, y, z]
        
    Returns:
        T: 4x4 transformation matrix
           [[R11, R12, R13, tx],
            [R21, R22, R23, ty],
            [R31, R32, R33, tz],
            [ 0,   0,   0,  1]]
    """
    T = np.eye(4)
    T[0:3, 0:3] = quat_to_rotation_matrix(quat)  # 회전 부분
    T[0:3, 3] = pos                              # 평행이동 부분
    return T

def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):
    """
    Sim-to-Sim Deploy 핵심 함수: 앵커링을 통한 상대 변환 계산
    
    이 함수는 논문의 ξ_{b_anchor} 계산에 핵심적인 역할을 합니다.
    Isaac Lab의 subtract_frame_transforms와 동일한 수학적 원리를 구현하여
    좌표계 변환 없이도 모션 트래킹이 가능하도록 합니다.
    
    === 수학적 배경 ===
    - T_A: Robot frame의 transformation matrix (현재 로봇 상태)
    - T_B: Mocap frame의 transformation matrix (목표 모션 상태)
    - T_rel = T_A^(-1) * T_B: Robot 기준에서 Mocap의 상대 변환
    
    === 물리적 의미 ===
    "로봇을 기준으로 목표 모션이 어디에/어떻게 위치하는가?"
    이 상대 변환을 통해 좌표계 차이를 흡수하고 모션 트래킹을 수행합니다.
    
    === 논문과의 연관성 ===
    논문의 Observation 구성: o = [c, ξ_{b_anchor}, V_{b_root}, q_joint,r, v_joint,r, a_last]
    이 함수는 ξ_{b_anchor} ∈ ℝ^9 (3+6) 계산에 사용됩니다.
    
    === Sim-to-Sim에서의 역할 ===
    1. 좌표계 독립성: MuJoCo vs Isaac Lab 좌표계 차이 흡수
    2. 앵커링: 로봇과 모션 데이터 간의 상대적 정렬 유지
    3. 정규화: 절대 좌표계 대신 상대적 관계에 집중
    
    Args:
        pos_a: 로봇 앵커 바디의 현재 위치 (3,) [x, y, z]
        quat_a: 로봇 앵커 바디의 현재 자세 (4,) [w, x, y, z]
        pos_b: 모션 데이터의 앵커 바디 위치 (3,) [x, y, z]  
        quat_b: 모션 데이터의 앵커 바디 자세 (4,) [w, x, y, z]
        
    Returns:
        rel_pos: 로봇 기준 모션의 상대 위치 (3,) - 논문의 ξ_{b_anchor} 위치 부분
        rel_quat: 로봇 기준 모션의 상대 회전 (4,) - 논문의 ξ_{b_anchor} 회전 부분
    """
    # 1. 4x4 transformation matrices 생성
    T_A = pose_to_transformation_matrix(pos_a, quat_a)  # Robot frame
    T_B = pose_to_transformation_matrix(pos_b, quat_b)  # Mocap frame
    
    # 2. 상대 변환 계산: T_rel = T_A^(-1) * T_B
    T_A_inv = np.linalg.inv(T_A)  # Robot frame의 역변환
    T_rel = T_A_inv @ T_B         # 상대 변환 행렬
    
    # 3. 결과 추출
    rel_pos = T_rel[0:3, 3]       # 상대 위치 (translation part)
    rel_rotation = T_rel[0:3, 0:3] # 상대 회전 (rotation part)
    
    # 4. 회전 행렬을 쿼터니언으로 변환
    rel_quat: np.ndarray = rotation_matrix_to_quaternion(rel_rotation)
    
    return rel_pos, rel_quat

def rotation_matrix_to_quaternion(R):
    """3x3 회전 행렬을 쿼터니언으로 변환 [w, x, y, z]"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    quat = np.array([qw, qx, qy, qz])
    return quat / np.linalg.norm(quat)  # 정규화



def pd_control(target_q, current_q, kp, target_dq, current_dq, kd):
    """Calculates torques from position commands"""
    return (target_q - current_q) * kp + (target_dq - current_dq) * kd




if __name__ == "__main__":
    """
    === Sim-to-Sim Deploy 메인 실행부 ===
    
    이 섹션에서는 논문의 Beyond Mimic 방법론을 구현한 
    sim-to-sim 배포 시스템의 전체 파이프라인을 실행합니다.
    """
    
    # =============================================================================
    # 1. 시뮬레이션 환경 설정
    # =============================================================================
    xml_path = "../source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
    simulation_duration = 60.0                                             # 시뮬레이션 총 시간 (초) - 테스트용
    simulation_dt = 0.005                                                   # Isaac Lab과 동일한 시뮬레이션 타임스텝 (0.005초 = 200Hz)
    control_decimation = 4                                                  # Isaac Lab과 동일한 제어기 업데이트 주파수 (simulation_dt * control_decimation = 0.02초; 50Hz)
    
    # =============================================================================
    # 2. 모션 데이터 로드 (Isaac Lab에서 export된 NPZ 파일)
    # =============================================================================
    motion_file = "../artifacts/dance2_subject5:v0/motion.npz"
    mocap =  np.load(motion_file)
    mocap_pos = mocap["body_pos_w"]        # 논문의 Reference Motion 위치 데이터
    mocap_quat = mocap["body_quat_w"]      # 논문의 Reference Motion 자세 데이터
    mocap_joint_pos = mocap["joint_pos"]   # 논문의 c = [q_joint,m, v_joint,m] 중 관절 위치 부분
    mocap_joint_vel = mocap["joint_vel"]   # 논문의 c = [q_joint,m, v_joint,m] 중 관절 속도 부분
    
    # =============================================================================
    # 3. 학습된 정책 로드 (Isaac Lab에서 export된 ONNX 모델)
    # =============================================================================
    policy_path = "../logs/rsl_rl/g1_flat/2025-10-01_11-31-29_run_test22/exported/policy.onnx"
    num_actions = 29    # 29개의 관절 조절 (G1 로봇의 관절 수)
    num_obs = 160  # ONNX 모델이 기대하는 관찰값 차원 : 160차원
    
    # =============================================================================
    # 4. Sim-to-Sim 호환성을 위한 관절 순서 매핑
    # =============================================================================
    # MuJoCo는 XML 파일의 순서대로 관절을 인덱싱하므로, Isaac Lab과 MuJoCo 간의
    # 관절 순서 차이를 해결하기 위한 매핑이 필요합니다.
    # 잘못된 매핑 시 제어 신호가 꼬일 수 있습니다 (팔 토크가 다리 토크로 적용 등)
    mujoco_joint_seq = [
            "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    ]


    # Isaac Lab에서 실제 body 순서 (30개) - debug_body_indices.py로 확인됨
    isaac_body_names = [
        "pelvis",                    # 0
        "left_hip_pitch_link",       # 1
        "right_hip_pitch_link",      # 2
        "waist_yaw_link",           # 3
        "left_hip_roll_link",       # 4
        "right_hip_roll_link",      # 5
        "waist_roll_link",          # 6
        "left_hip_yaw_link",        # 7
        "right_hip_yaw_link",       # 8
        "torso_link",               # 9 ← NPZ 파일의 anchor body
        "left_knee_link",           # 10
        "right_knee_link",          # 11
        "left_shoulder_pitch_link", # 12
        "right_shoulder_pitch_link",# 13
        "left_ankle_pitch_link",    # 14
        "right_ankle_pitch_link",   # 15
        "left_shoulder_roll_link",  # 16
        "right_shoulder_roll_link", # 17
        "left_ankle_roll_link",     # 18
        "right_ankle_roll_link",    # 19
        "left_shoulder_yaw_link",   # 20
        "right_shoulder_yaw_link",  # 21
        "left_elbow_link",          # 22
        "right_elbow_link",         # 23
        "left_wrist_roll_link",     # 24
        "right_wrist_roll_link",    # 25
        "left_wrist_pitch_link",    # 26
        "right_wrist_pitch_link",   # 27
        "left_wrist_yaw_link",      # 28
        "right_wrist_yaw_link",     # 29
    ]
    

    # =============================================================================
    # 5. ONNX 모델 메타데이터 파싱 (Sim-to-Sim 호환성 확보)
    # =============================================================================
    # Isaac Lab에서 export된 ONNX 모델의 메타데이터를 읽어서 MuJoCo 환경의 설정을 동기화합니다.
    # 이를 통해 sim2sim 변환을 달성하고 정책이 올바르게 동작하도록 합니다.
    rl_model: onnx.ModelProto = onnx.load(policy_path)
    
    # Isaac Lab에서 RL 정책 훈련을 isaac_joint_seq 순서로 학습했으므로,
    # MuJoCo에서 실행할 때는 mujoco_joint_seq(g1.xml) 순서로 변환해야 합니다.
    for prop in rl_model.metadata_props:
        if prop.key == "joint_names":
            # Isaac Lab에서 학습된 정책이 사용하는 관절 순서 (29개)
            # 논문의 q_joint,r, v_joint,r 계산 시 이 순서를 따라야 합니다.
            isaac_joint_seq: list[str] = prop.value.split(",")
            
        if prop.key == "default_joint_pos":  
            # Isaac Lab에서 사용한 기본 관절 위치 (중립 자세)
            # 논문의 q_joint,r 계산 시 상대값을 구하기 위해 사용됩니다.
            isaac_joint_pos_array = np.array([float(x) for x in prop.value.split(",")])
            # MuJoCo 순서로 변환 (Sim-to-Sim 호환성)
            mujoco_joint_pos_array = np.array([isaac_joint_pos_array[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq])
            
        if prop.key == "joint_stiffness":
            # PD 제어기에서 사용할 관절 강성 계수
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
            stiffness_array = np.array([stiffness_array_seq[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq])
            
        if prop.key == "joint_damping":
            # PD 제어기에서 사용할 관절 감쇠 계수
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
            damping_array = np.array([damping_array_seq[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq])
        
        if prop.key == "action_scale":
            # 정책 출력을 실제 관절 위치로 변환하는 스케일 팩터
            # 논문의 액션 스케일링에 해당합니다.
            action_scale = np.array([float(x) for x in prop.value.split(",")])
            
        print(f"{prop.key}: {prop.value}")
    # =============================================================================
    # 6. 시뮬레이션 및 정책 초기화
    # =============================================================================
    # 논문의 observation 구성에 맞는 배열 초기화
    action: np.ndarray = np.zeros(num_actions, dtype=np.float32)  # 정책 출력 (29차원)
    obs: np.ndarray = np.zeros(num_obs, dtype=np.float32)        # 논문의 o = [c, ξ_{b_anchor}, V_{b_root}, q_joint,r, v_joint,r, a_last] (160차원)

    # MuJoCo 물리 시뮬레이션 환경 로드
    mj_model = mujoco.MjModel.from_xml_path(xml_path)      # 물리 시뮬레이션 환경 정의
    mj_data = mujoco.MjData(mj_model)                     # 물리 시뮬레이션 상태 관리
    mj_model.opt.timestep = simulation_dt                 # Isaac Lab과 동일한 타임스텝 설정

    # Isaac Lab에서 export된 ONNX 정책 로드
    policy = onnxruntime.InferenceSession(policy_path)
    # ONNX 정책 입력/출력 이름 (사용되지 않지만 참고용으로 유지)

    # 정책 메모리 역할을 하는 이전 액션 버퍼 (논문의 a_last)
    action_buffer: np.ndarray = np.zeros((num_actions,), dtype=np.float32)  

    timestep = 0
    anchor_body_name = "torso_link"
    mocap_anchor_body_index = isaac_body_names.index(anchor_body_name)  # Isaac Lab에서는 9
    # 초기 모션 데이터 (실제로는 루프 내에서 업데이트됨)
    target_dof_pos = mujoco_joint_pos_array.copy()             # 시뮬레이터가 (시작했을때 초기 관절 위치 배열을 mujoco_joint_pos_array에 저장
    
    mj_data.qpos[7:] = target_dof_pos                         # anchor body(torso)에 한해서는  $\hat T_{b_{anchor,r}}$ 와  $T_{b_{anchor,m}}$ 이 개념적으로 같다                       
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name) # /home/keti/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml 에서 16번째 body의 이름은 torso_link 이다.
    if body_id == -1:
        raise ValueError(f"Body {anchor_body_name} not found in model")
    
    
    counter = 0 # 제어 신호 적용 횟수
    
    # 트래킹 성능 로깅을 위한 변수들
    log_interval = 100  # 100 스텝마다 로깅
    
    # 대표 body들 정의 (다리와 팔)
    representative_bodies = {
        'left_ankle': 'left_ankle_roll_link',     # 왼발
        'right_ankle': 'right_ankle_roll_link',   # 오른발
        'left_hand': 'left_wrist_yaw_link',       # 왼손
        'right_hand': 'right_wrist_yaw_link'      # 오른손
    }
    
    # 대표 body들의 MuJoCo body ID 찾기
    representative_body_ids = {}
    for key, body_name in representative_bodies.items():
        body_id_rep = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id_rep != -1:
            representative_body_ids[key] = body_id_rep
            print(f"Found {key} ({body_name}): body_id = {body_id_rep}")
        else:
            print(f"Warning: {key} ({body_name}) not found in model")
    
    # 대표 body들의 Isaac Lab 인덱스 찾기
    representative_isaac_indices = {}
    for key, body_name in representative_bodies.items():
        if body_name in isaac_body_names:
            representative_isaac_indices[key] = isaac_body_names.index(body_name)
        else:
            print(f"Warning: {key} ({body_name}) not found in Isaac Lab body names")
    
    # 성능 지표 (commands.py 기반)
    additional_metrics = {
        'error_anchor_pos': [],
        'error_anchor_rot': [],
        'error_joint_pos': [],
        'error_joint_vel': [],
        'error_body_pos': [],
        'error_body_rot': [],
        'error_body_lin_vel': [],
        'error_body_ang_vel': []
    }
    
    # =============================================================================
    # 7. Sim-to-Sim Deploy 메인 루프 실행
    # =============================================================================
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # =============================================================================
            # 7.1 물리 시뮬레이션 스텝 실행 (200Hz)
            # =============================================================================
            mujoco.mj_step(mj_model, mj_data)  # MuJoCo 물리 시뮬레이션 진행
            
            # =============================================================================
            # 7.2 PD 제어기를 통한 관절 토크 계산 및 적용
            # =============================================================================
            # 정책에서 출력된 목표 관절 위치를 PD 제어기로 토크 변환
            tau = pd_control(
                target_q=target_dof_pos,           # 정책이 출력한 목표 관절 위치
                current_q=mj_data.qpos[7:],        # 현재 관절 위치
                kp=stiffness_array,                # 관절 강성 계수
                target_dq=np.zeros_like(damping_array),  # 목표 관절 속도 (0으로 설정)
                current_dq=mj_data.qvel[6:],       # 현재 관절 속도
                kd=damping_array                   # 관절 감쇠 계수
            )
            mj_data.ctrl[:] = tau  # 계산된 토크를 액추에이터에 적용
            
            counter += 1
            # =============================================================================
            # 7.3 정책 추론 및 관찰값 계산 (50Hz - control_decimation=4)
            # =============================================================================
            if counter % control_decimation == 0:
                # =============================================================================
                # 7.3.1 현재 로봇 상태 및 목표 모션 데이터 추출
                # =============================================================================
                robot_anchor_pos: np.ndarray = mj_data.xpos[body_id]              # 현재 로봇 앵커 바디 위치 (torso_link)
                robot_anchor_quat: np.ndarray = mj_data.xquat[body_id]           # 현재 로봇 앵커 바디 자세 (torso_link)
                
                # 논문의 c = [q_joint,m, v_joint,m] 구성 (Reference Motion)
                mocap_input = np.concatenate((mocap_joint_pos[timestep,:],mocap_joint_vel[timestep,:]),axis=0)    # shape : (58,)
                
                # 목표 모션의 앵커 바디 상태
                mocap_anchor_pos = mocap_pos[timestep, mocap_anchor_body_index, :]  # 목표 모션 앵커 바디 위치
                mocap_anchor_quat = mocap_quat[timestep, mocap_anchor_body_index, :]  # 목표 모션 앵커 바디 자세
                
                # =============================================================================
                # 7.3.2 앵커링을 통한 상대 변환 계산 (논문의 ξ_{b_anchor})
                # =============================================================================
                # Sim-to-Sim 핵심: 좌표계 변환 없이 상대적 관계 계산
                # anchor_pos_track_erro : 논문의 ξ_{b_anchor} 위치 부분
                # anchor_quat_track_error : 논문의 ξ_{b_anchor} 회전 부분
                anchor_pos_track_error, anchor_quat_track_error = subtract_frame_transforms_mujoco(
                    pos_a=robot_anchor_pos,    # 로봇 기준
                    quat_a=robot_anchor_quat,  # 로봇 기준
                    pos_b=mocap_anchor_pos,    # 모션 기준
                    quat_b=mocap_anchor_quat   # 모션 기준
                )
                
                # 회전 행렬을 6차원 벡터로 변환 (논문의 ξ_{b_anchor} 회전 부분)
                anchor_ori = np.zeros(9)
                mujoco.mju_quat2Mat(anchor_ori, anchor_quat_track_error)
                anchor_ori = anchor_ori.reshape(3, 3)[:, :2]  # 첫 2열만 사용 (6차원)
                anchor_ori = anchor_ori.reshape(-1,)
                # =============================================================================
                # 7.3.3 논문의 Observation 구성 구현
                # =============================================================================
                # 논문의 o = [c, ξ_{b_anchor}, V_{b_root}, q_joint,r, v_joint,r, a_last] 구현
                # Sim-to-Sim 핵심: 좌표계 변환 없이도 작동하는 상대적 관찰값 사용
                
                offset = 0
                
                # 1. c ∈ ℝ^58 : Reference Motion의 관절 위치 및 속도 (29+29)
                obs[offset:offset + 58] = mocap_input       # 논문의 c = [q_joint,m, v_joint,m]
                offset += 58
                
                # 2. ξ_{b_anchor} ∈ ℝ^9 : Anchor Body의 자세 추적 오차 (3+6)
                obs[offset:offset + 3] = anchor_pos_track_error         # 논문의 ξ_{b_anchor} 위치 부분 (3차원)
                offset += 3
                obs[offset:offset + 6] = anchor_ori                    # 논문의 ξ_{b_anchor} 회전 부분 (6차원)
                offset += 6
                
                # 3. V_{b_root} ∈ ℝ^6 : Robot's root twist expressed in root frame (3+3)
                obs[offset:offset + 3] = mj_data.qvel[0:3]             # 베이스 선형 속도 (3차원)
                offset += 3
                obs[offset:offset + 3] = mj_data.qvel[3 : 6]          # 베이스 각속도 (3차원)
                offset += 3
                
                # 4. q_joint,r ∈ ℝ^29 : 로봇의 모든 Joint의 현재 각도 (절대값)
                # 논문에서는 절대 관절 위치를 사용하므로 기본 관절 위치를 빼지 않습니다.
                qpos_xml = mj_data.qpos[7 : 7 + num_actions]  # MuJoCo XML 순서의 관절 위치
                qpos_seq = np.array([qpos_xml[mujoco_joint_seq.index(joint)] for joint in isaac_joint_seq])
                obs[offset:offset + num_actions] = qpos_seq - isaac_joint_pos_array# 논문의 q_joint,r (절대값)
                offset += num_actions
                
                # 5. v_joint,r ∈ ℝ^29 : 로봇의 모든 Joint의 현재 각속도 (절대값)
                qvel_xml = mj_data.qvel[6 : 6 + num_actions]  # MuJoCo XML 순서의 관절 속도
                qvel_seq = np.array([qvel_xml[mujoco_joint_seq.index(joint)] for joint in isaac_joint_seq])
                obs[offset:offset + num_actions] = qvel_seq  # 논문의 v_joint,r (절대값)
                offset += num_actions   
                
                # 6. a_last ∈ ℝ^29 : Policy가 직전에 취한 행동 (메모리 역할)
                obs[offset:offset + num_actions] = action_buffer  # 논문의 a_last (정책 메모리)

                # =============================================================================
                # 7.3.4 ONNX 정책 추론 실행
                # =============================================================================
                # Isaac Lab에서 학습된 정책을 MuJoCo에서 실행
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # 배치 차원 추가
                action = policy.run(['actions'], {
                    'obs': obs_tensor.numpy(),
                    'time_step': np.array([timestep], dtype=np.float32).reshape(1,1)
                })[0]
                action = np.asarray(action).reshape(-1)  # 정책 출력 (29차원)
                action_buffer = action.copy()  # 다음 스텝을 위한 메모리 저장
                
                # =============================================================================
                # 7.3.5 정책 출력을 실제 관절 위치로 변환
                # =============================================================================
                # 논문의 액션 스케일링: q_{j,t} = α_j * a_{j,t} + q̄_j
                # α_j: action_scale, a_{j,t}: 정책 출력, q̄_j: 기본 관절 위치
                target_dof_pos = action * action_scale + isaac_joint_pos_array
                target_dof_pos = target_dof_pos.reshape(-1,)
                # Isaac Lab 순서에서 MuJoCo 순서로 변환 (Sim-to-Sim 호환성)
                target_dof_pos = np.array([target_dof_pos[isaac_joint_seq.index(joint)] for joint in mujoco_joint_seq])
                
                # =============================================================================
                # 7.3.6 성능 지표 계산 (commands.py 기반)
                # =============================================================================
                # Isaac Lab의 MotionCommand 클래스와 동일한 메트릭을 계산합니다.
                
                # 관절 데이터 수집 (Isaac Lab 순서로 변환)
                current_joint_pos = mj_data.qpos[7:]  # 현재 관절 위치 (MuJoCo 순서)
                target_joint_pos_isaac = mocap_joint_pos[timestep, :]  # 목표 관절 위치 (Isaac 순서)
                current_joint_pos_isaac = np.array([current_joint_pos[mujoco_joint_seq.index(joint)] for joint in isaac_joint_seq])
                
                current_joint_vel = mj_data.qvel[6:]  # 현재 관절 속도 (MuJoCo 순서)
                target_joint_vel_isaac = mocap_joint_vel[timestep, :]  # 목표 관절 속도 (Isaac 순서)
                current_joint_vel_isaac = np.array([current_joint_vel[mujoco_joint_seq.index(joint)] for joint in isaac_joint_seq])
                
                # 바디 부위 데이터 수집 (대표 body들만)
                robot_body_pos = np.array([mj_data.xpos[representative_body_ids[key]] for key in representative_bodies.keys() 
                                         if key in representative_body_ids])
                mocap_body_pos = np.array([mocap_pos[timestep, representative_isaac_indices[key], :] for key in representative_bodies.keys() 
                                         if key in representative_isaac_indices])
                robot_body_quat = np.array([mj_data.xquat[representative_body_ids[key]] for key in representative_bodies.keys() 
                                          if key in representative_body_ids])
                mocap_body_quat = np.array([mocap_quat[timestep, representative_isaac_indices[key], :] for key in representative_bodies.keys() 
                                          if key in representative_isaac_indices])
                
                # 바디 속도 데이터 (간단히 0으로 설정 - 실제로는 이전 프레임과의 차이로 계산 가능)
                robot_body_lin_vel = np.zeros_like(robot_body_pos)
                mocap_body_lin_vel = np.zeros_like(mocap_body_pos)
                robot_body_ang_vel = np.zeros_like(robot_body_pos)
                mocap_body_ang_vel = np.zeros_like(mocap_body_pos)
                
                # 성능 지표 계산 (commands.py 기반)
                additional_metrics_step = calculate_additional_metrics(
                    robot_anchor_pos_w=robot_anchor_pos,
                    robot_anchor_quat=robot_anchor_quat,
                    mocap_anchor_pos_w=mocap_anchor_pos,
                    mocap_anchor_quat=mocap_anchor_quat,
                    robot_joint_pos=current_joint_pos_isaac,
                    mocap_joint_pos=target_joint_pos_isaac,
                    robot_joint_vel=current_joint_vel_isaac,
                    mocap_joint_vel=target_joint_vel_isaac,
                    robot_body_pos=robot_body_pos,
                    mocap_body_pos=mocap_body_pos,
                    robot_body_quat=robot_body_quat,
                    mocap_body_quat=mocap_body_quat,
                    robot_body_lin_vel_w=robot_body_lin_vel,
                    mocap_body_lin_vel_w=mocap_body_lin_vel,
                    robot_body_ang_vel_w=robot_body_ang_vel,
                    mocap_body_ang_vel_w=mocap_body_ang_vel
                )
                
                # 성능 지표 저장
                for key, value in additional_metrics_step.items():
                    additional_metrics[key].append(value)
                
                # 실시간 로깅 출력 (commands.py 기반 지표 사용)
                if timestep % log_interval == 0:
                    print(f"\n=== 트래킹 성능 리포트 (Step {timestep}) ===")
                    print(f"Anchor Position Error: {additional_metrics_step['error_anchor_pos']:.4f} m")
                    print(f"Anchor Rotation Error: {additional_metrics_step['error_anchor_rot']:.4f} rad")
                    print(f"Joint Position Error: {additional_metrics_step['error_joint_pos']:.4f} rad")
                    print(f"Joint Velocity Error: {additional_metrics_step['error_joint_vel']:.4f} rad/s")
                    
                    if 'error_body_pos' in additional_metrics_step:
                        print(f"Body Position Error: {additional_metrics_step['error_body_pos']:.4f} m")
                        print(f"Body Rotation Error: {additional_metrics_step['error_body_rot']:.4f} rad")
                    
                    # 최근 100스텝 평균 성능
                    if len(additional_metrics['error_anchor_pos']) >= log_interval:
                        recent_anchor_pos = np.mean(additional_metrics['error_anchor_pos'][-log_interval:])
                        recent_anchor_rot = np.mean(additional_metrics['error_anchor_rot'][-log_interval:])
                        recent_joint_pos = np.mean(additional_metrics['error_joint_pos'][-log_interval:])
                        recent_joint_vel = np.mean(additional_metrics['error_joint_vel'][-log_interval:])
                        
                        print(f"\n최근 {log_interval}스텝 평균:")
                        print(f"   Anchor Position: {recent_anchor_pos:.4f} m")
                        print(f"   Anchor Rotation: {recent_anchor_rot:.4f} rad")
                        print(f"   Joint Position: {recent_joint_pos:.4f} rad")
                        print(f"   Joint Velocity: {recent_joint_vel:.4f} rad/s")
                        
                        if 'error_body_pos' in additional_metrics and len(additional_metrics['error_body_pos']) >= log_interval:
                            recent_body_pos = np.mean(additional_metrics['error_body_pos'][-log_interval:])
                            recent_body_rot = np.mean(additional_metrics['error_body_rot'][-log_interval:])
                            print(f"   Body Position: {recent_body_pos:.4f} m")
                            print(f"   Body Rotation: {recent_body_rot:.4f} rad")
                        
                        # # 성능 등급 표시 (commands.py 기준)
                        # if recent_anchor_pos < 0.01 and recent_anchor_rot < 0.1:
                        #     print("트래킹 성능: 우수 (Excellent)")
                        # elif recent_anchor_pos < 0.05 and recent_anchor_rot < 0.3:
                        #     print("트래킹 성능: 양호 (Good)")
                        # else:
                        #     print("트래킹 성능: 개선 필요 (Needs Improvement)")
                
                timestep+=1
                

            # =============================================================================
            # 7.4 시뮬레이션 동기화 및 시간 관리
            # =============================================================================
            viewer.sync()   # MuJoCo 뷰어와 시뮬레이션 데이터 동기화
            
            # Isaac Lab과 동일한 시뮬레이션 속도 유지
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # =============================================================================
    # 8. Sim-to-Sim Deploy 성능 요약 및 분석 (commands.py 기반)
    # =============================================================================
    print("\n" + "="*60)
    print("Sim-to-Sim Deploy 완료 - Beyond Mimic 성능 요약 (commands.py 기반)")
    print("="*60)
    
    if additional_metrics['error_anchor_pos']:
        # commands.py 기반 핵심 성능 지표 계산
        avg_anchor_pos_error = np.mean(additional_metrics['error_anchor_pos'])
        avg_anchor_rot_error = np.mean(additional_metrics['error_anchor_rot'])
        avg_joint_pos_error = np.mean(additional_metrics['error_joint_pos'])
        avg_joint_vel_error = np.mean(additional_metrics['error_joint_vel'])
        
        max_anchor_pos_error = np.max(additional_metrics['error_anchor_pos'])
        max_anchor_rot_error = np.max(additional_metrics['error_anchor_rot'])
        
        print(f"commands.py 기반 핵심 성능 지표:")
        print(f"   Anchor Position Error: {avg_anchor_pos_error:.4f} m (최대: {max_anchor_pos_error:.4f} m)")
        print(f"   Anchor Rotation Error: {avg_anchor_rot_error:.4f} rad (최대: {max_anchor_rot_error:.4f} rad)")
        print(f"   Joint Position Error: {avg_joint_pos_error:.4f} rad")
        print(f"   Joint Velocity Error: {avg_joint_vel_error:.4f} rad/s")
        
        # 바디 부위 성능 (있는 경우)
        if 'error_body_pos' in additional_metrics and additional_metrics['error_body_pos']:
            avg_body_pos_error = np.mean(additional_metrics['error_body_pos'])
            avg_body_rot_error = np.mean(additional_metrics['error_body_rot'])
            print(f"\nBody Part Performance:")
            print(f"   Body Position Error: {avg_body_pos_error:.4f} m")
            print(f"   Body Rotation Error: {avg_body_rot_error:.4f} rad")
        
        print(f"\nSim-to-Sim 실행 통계:")
        print(f"   총 처리된 스텝: {len(additional_metrics['error_anchor_pos'])}")
        print(f"   시뮬레이션 시간: {len(additional_metrics['error_anchor_pos']) * simulation_dt:.2f}초")
        print(f"   정책 추론 주파수: {1/(simulation_dt * control_decimation):.1f}Hz")
        
        # =============================================================================
        # Sim-to-Sim 성공도 평가 (commands.py 기준)
        # =============================================================================
        if avg_anchor_pos_error < 0.01 and avg_anchor_rot_error < 0.1:
            print("\n🎉 Sim-to-Sim 성공도: 우수 (Excellent)")
            print("   Beyond Mimic 방법론이 성공적으로 구현되었습니다!")
            print("   Isaac Lab → MuJoCo 전환이 매우 정확하게 수행되었습니다.")
        elif avg_anchor_pos_error < 0.05 and avg_anchor_rot_error < 0.3:
            print("\n✅ Sim-to-Sim 성공도: 양호 (Good)")
            print("   모션 트래킹이 잘 수행되고 있지만 개선 여지가 있습니다.")
            print("   좌표계 변환 없이도 상당한 성능을 달성했습니다.")
        else:
            print("\n⚠️  Sim-to-Sim 성공도: 개선 필요 (Needs Improvement)")
            print("   정책 튜닝이나 학습 데이터 개선이 필요할 수 있습니다.")
            print("   앵커링 메커니즘이나 관찰값 구성 재검토를 권장합니다.")
    else:
        print("❌ 경고: 성능 데이터가 기록되지 않았습니다.")
        print("   시뮬레이션이 정상적으로 실행되지 않았을 수 있습니다.")
    
    # =============================================================================
    # 9. 성능 플롯 생성 및 저장
    # =============================================================================
    print("\n" + "="*60)
    print("성능 플롯 생성 중...")
    print("="*60)
    
    # 성능 플롯 저장 (commands.py 기반 지표만)
    save_performance_plots(
        additional_metrics, 
        save_dir="./performance_plots",
        simulation_dt=simulation_dt,
        control_decimation=control_decimation
    )
    
    print("="*60)
    print("Beyond Mimic Sim-to-Sim Deploy 완료")
    print("="*60)
