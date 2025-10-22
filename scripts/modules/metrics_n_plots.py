import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def quat_error_magnitude(q1, q2):
    """쿼터니언 간의 각도 차이를 계산합니다 (Isaac Lab의 quat_error_magnitude와 동일)"""
    # 쿼터니언 내적을 통한 각도 차이 계산
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, 0.0, 1.0)  # 수치 안정성 확보
    return 2 * np.arccos(dot_product)



def calculate_additional_metrics(robot_anchor_pos_w, robot_anchor_quat, mocap_anchor_pos_w, mocap_anchor_quat,
                                robot_joint_pos, mocap_joint_pos, robot_joint_vel, mocap_joint_vel,
                                robot_body_pos, mocap_body_pos, robot_body_quat, mocap_body_quat,
                                robot_body_lin_vel_w, mocap_body_lin_vel_w, robot_body_ang_vel_w, mocap_body_ang_vel_w):
    """
    commands.py 기반 추가 성능 지표를 계산합니다.
    
    Args:
        robot_anchor_pos: 로봇 앵커 위치 (3,)
        robot_anchor_quat: 로봇 앵커 쿼터니언 (4,)
        mocap_anchor_pos: 목표 앵커 위치 (3,)
        mocap_anchor_quat: 목표 앵커 쿼터니언 (4,)
        robot_joint_pos: 로봇 관절 위치 (29,)
        mocap_joint_pos: 목표 관절 위치 (29,)
        robot_joint_vel: 로봇 관절 속도 (29,)
        mocap_joint_vel: 목표 관절 속도 (29,)
        robot_body_pos: 로봇 바디 위치 (num_bodies, 3)
        mocap_body_pos: 목표 바디 위치 (num_bodies, 3)
        robot_body_quat: 로봇 바디 쿼터니언 (num_bodies, 4)
        mocap_body_quat: 목표 바디 쿼터니언 (num_bodies, 4)
        robot_body_lin_vel: 로봇 바디 선형 속도 (num_bodies, 3)
        mocap_body_lin_vel: 목표 바디 선형 속도 (num_bodies, 3)
        robot_body_ang_vel: 로봇 바디 각속도 (num_bodies, 3)
        mocap_body_ang_vel: 목표 바디 각속도 (num_bodies, 3)
    
    Returns:
        dict: 추가 성능 지표들
    """
    metrics = {}
    
    # 1. 앵커 바디 추적 성능 (commands.py 기반)
    metrics['error_anchor_pos'] = np.linalg.norm(robot_anchor_pos_w - mocap_anchor_pos_w)
    metrics['error_anchor_rot'] = quat_error_magnitude(robot_anchor_quat, mocap_anchor_quat)
    
    # 2. 관절 추적 성능 (commands.py 기반)
    metrics['error_joint_pos'] = np.linalg.norm(robot_joint_pos - mocap_joint_pos)
    metrics['error_joint_vel'] = np.linalg.norm(robot_joint_vel - mocap_joint_vel)
    
    # 3. 바디 부위 추적 성능 (commands.py 기반)
    if robot_body_pos is not None and mocap_body_pos is not None:
        metrics['error_body_pos'] = np.mean(np.linalg.norm(robot_body_pos - mocap_body_pos, axis=-1))
        metrics['error_body_rot'] = np.mean([quat_error_magnitude(robot_body_quat[i], mocap_body_quat[i]) 
                                           for i in range(len(robot_body_quat))])
        metrics['error_body_lin_vel'] = np.mean(np.linalg.norm(robot_body_lin_vel_w - mocap_body_lin_vel_w, axis=-1))
        metrics['error_body_ang_vel'] = np.mean(np.linalg.norm(robot_body_ang_vel_w - mocap_body_ang_vel_w, axis=-1))
    
    return metrics

def save_performance_plots(additional_metrics, save_dir="/home/keti/whole_body_tracking/scripts/Beyond_mimic_sim2sim_G1/performance_plots", simulation_dt=0.005, control_decimation=4):
    """
    commands.py 기반 성능 지표들을 시각화하고 저장합니다.
    
    Args:
        additional_metrics: commands.py 기반 성능 지표 데이터
        save_dir: 저장할 디렉토리
        simulation_dt: 시뮬레이션 타임스텝 (초)
        control_decimation: 제어기 업데이트 주파수 (시뮬레이션 스텝 대비)
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 시간 축 계산 (제어기 업데이트 주파수 기준)
    control_dt = simulation_dt * control_decimation  # 0.005 * 4 = 0.02초 (50Hz)
    if additional_metrics and 'error_anchor_pos' in additional_metrics:
        num_steps = len(additional_metrics['error_anchor_pos'])
        time_axis = np.arange(num_steps) * control_dt  # 시간 축 (초)
    else:
        time_axis = None
    
    # 1. 앵커 및 관절 성능 지표 플롯
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sim-to-Sim Deploy Performance Metrics (commands.py based)', fontsize=16)
    
    # 앵커 위치 오차
    if 'error_anchor_pos' in additional_metrics and additional_metrics['error_anchor_pos']:
        axes[0, 0].plot(time_axis, additional_metrics['error_anchor_pos'], 'b-', linewidth=1)
        axes[0, 0].set_title('Anchor Position Error')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Error (m)')
        axes[0, 0].grid(True)
    
    # 앵커 회전 오차
    if 'error_anchor_rot' in additional_metrics and additional_metrics['error_anchor_rot']:
        axes[0, 1].plot(time_axis, additional_metrics['error_anchor_rot'], 'r-', linewidth=1)
        axes[0, 1].set_title('Anchor Rotation Error')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Error (rad)')
        axes[0, 1].grid(True)
    
    # 관절 위치 오차
    if 'error_joint_pos' in additional_metrics and additional_metrics['error_joint_pos']:
        axes[1, 0].plot(time_axis, additional_metrics['error_joint_pos'], 'g-', linewidth=1)
        axes[1, 0].set_title('Joint Position Error')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Error (rad)')
        axes[1, 0].grid(True)
    
    # 관절 속도 오차
    if 'error_joint_vel' in additional_metrics and additional_metrics['error_joint_vel']:
        axes[1, 1].plot(time_axis, additional_metrics['error_joint_vel'], 'm-', linewidth=1)
        axes[1, 1].set_title('Joint Velocity Error')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Error (rad/s)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anchor_joint_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 바디 부위별 성능 플롯
    if ('error_body_pos' in additional_metrics and additional_metrics['error_body_pos'] and
        'error_body_rot' in additional_metrics and additional_metrics['error_body_rot']):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Body Part Tracking Performance (commands.py based)', fontsize=16)
        
        # 바디 위치 오차
        axes[0, 0].plot(time_axis, additional_metrics['error_body_pos'], 'b-', linewidth=1)
        axes[0, 0].set_title('Body Position Error')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Error (m)')
        axes[0, 0].grid(True)
        
        # 바디 회전 오차
        axes[0, 1].plot(time_axis, additional_metrics['error_body_rot'], 'r-', linewidth=1)
        axes[0, 1].set_title('Body Rotation Error')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Error (rad)')
        axes[0, 1].grid(True)
        
        # 바디 선형 속도 오차
        if 'error_body_lin_vel' in additional_metrics and additional_metrics['error_body_lin_vel']:
            axes[1, 0].plot(time_axis, additional_metrics['error_body_lin_vel'], 'g-', linewidth=1)
            axes[1, 0].set_title('Body Linear Velocity Error')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Error (m/s)')
            axes[1, 0].grid(True)
        
        # 바디 각속도 오차
        if 'error_body_ang_vel' in additional_metrics and additional_metrics['error_body_ang_vel']:
            axes[1, 1].plot(time_axis, additional_metrics['error_body_ang_vel'], 'm-', linewidth=1)
            axes[1, 1].set_title('Body Angular Velocity Error')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Error (rad/s)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/body_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 성능 플롯이 저장되었습니다: {save_dir}")
    print(f"   - 앵커/관절 성능: anchor_joint_metrics_{timestamp}.png")
    if ('error_body_pos' in additional_metrics and additional_metrics['error_body_pos']):
        print(f"   - 바디 부위 성능: body_metrics_{timestamp}.png")

