"""
Custom RSL-RL runner with metrics logging functionality.
"""

import os
import statistics
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch

from rsl_rl.runners import OnPolicyRunner
from metrics_logger import MetricsLogger

class CustomOnPolicyRunner(OnPolicyRunner):
    """
    Custom OnPolicyRunner that includes metrics logging functionality.
    Extends the base RSL-RL OnPolicyRunner to log detailed training metrics.
    """
    
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        """
        Initialize the custom runner with metrics logging.
        
        Args:
            env: The environment to train on
            train_cfg: Training configuration dictionary
            log_dir: Directory for logging (will be passed to metrics logger)
            device: Device to run on (cpu/cuda)
        """
        # Initialize parent class
        super().__init__(env, train_cfg, log_dir, device)
        
        # Initialize metrics logger - save to the same directory as RSL-RL logs
        self.metrics_logger = MetricsLogger(save_dir=log_dir)
        
        # Store training start time
        self.training_start_time = datetime.now()
        
        print(f"[INFO] Custom metrics logging initialized. Logs will be saved to: {log_dir}")
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        Enhanced learn method with metrics logging.
        
        Args:
            num_learning_iterations: Number of iterations to train
            init_at_random_ep_len: Whether to initialize at random episode length
        """
        print(f"[INFO] Starting training with custom metrics logging for {num_learning_iterations} iterations")
        
        try:
            # Run the actual training (부모 클래스 호출)
            super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            # Save final metrics
            csv_path = self.metrics_logger.save_to_csv()
            print(f"[INFO] Training complete. Metrics saved to: {csv_path}")
    
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """
        Override RSL-RL's log method to capture metrics and maintain original formatting.
        
        Args:
            locs: Dictionary containing all local variables from learn() method
            width: Width for formatting
            pad: Padding for formatting
        """
        # 🎯 먼저 메트릭 추출 및 저장
        self._extract_and_log_metrics(locs)
        
        # 🎯 그 다음 원본 RSL-RL 로그 출력 (동일한 형식 유지)
        super().log(locs, width, pad)
    
    def _extract_and_log_metrics(self, locs: Dict[str, Any]) -> None:
        """
        Extract the 6 key metrics from RSL-RL's locals() and log them.
        
        Args:
            locs: Dictionary containing all local variables from learn() method
        """
        # RSL-RL 소스코드 기반으로 정확한 메트릭 추출
        
        # 1. Mean action noise std (Line 356 in RSL-RL)
        mean_action_noise_std = self.alg.policy.action_std.mean().item()
        
        # 2. Loss 값들 (loss_dict에서)
        loss_dict = locs.get('loss_dict', {})
        mean_value_function_loss = loss_dict.get('value_function', 0.0)
        mean_surrogate_loss = loss_dict.get('surrogate', 0.0)
        mean_entropy_loss = loss_dict.get('entropy', 0.0)
        
        # 3. Mean reward and episode length (rewbuffer, lenbuffer에서)
        rewbuffer = locs.get('rewbuffer', [])
        lenbuffer = locs.get('lenbuffer', [])
        
        mean_reward = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else 0.0
        mean_episode_length = statistics.mean(lenbuffer) if len(lenbuffer) > 0 else 0.0
        
        # 4. 기타 유용한 메트릭들
        iteration = locs.get('it', 0)
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        fps = int(collection_size / (locs.get('collection_time', 1) + locs.get('learn_time', 1)))
        
        # 6개 핵심 메트릭 + 추가 정보
        metrics = {
            'iteration': iteration,
            'mean_action_noise_std': mean_action_noise_std,
            'mean_value_function_loss': mean_value_function_loss,
            'mean_surrogate_loss': mean_surrogate_loss,
            'mean_entropy_loss': mean_entropy_loss,
            'mean_reward': mean_reward,
            'mean_episode_length': mean_episode_length,
            'fps': fps,
            'collection_time': locs.get('collection_time', 0.0),
            'learn_time': locs.get('learn_time', 0.0),
            'total_timesteps': self.tot_timesteps,
            'time_elapsed_seconds': (datetime.now() - self.training_start_time).total_seconds(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 메트릭 로거에 저장 (메모리에 누적)
        self.metrics_logger.log_metrics(metrics)
        
        # 진행 상황 표시 (매 10 iteration마다)
        if iteration % 10 == 0:
            print(f"[CSV] Logged iteration {iteration}: Reward={mean_reward:.2f}, Loss={mean_value_function_loss:.2f}")