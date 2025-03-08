"""
2次元幾何音響シミュレーションプログラム
音源から放射される音響粒子の挙動を可視化し、反射データを記録します
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# シミュレーションパラメータ
ROOM_WIDTH = 10.0    # 部屋の幅（m）
ROOM_HEIGHT = 5.0    # 部屋の高さ（m）
SOURCE_POS = np.array([ROOM_WIDTH/2, ROOM_HEIGHT/2])  # 音源位置
NUM_BALLS = 1000      # ボールの数
BALL_SPEED = 0.1     # ボールの速度（m/ステップ）
MAX_STEPS = 500      # 最大ステップ数
MAX_REFLECTIONS = 5  # 最大反射回数

class SoundBall:
    def __init__(self, start_pos, direction):
        self.positions = [start_pos.copy()]  # 軌跡の記録
        self.direction = direction / np.linalg.norm(direction)  # 進行方向（正規化）
        self.reflection_count = 0
        self.is_active = True

    def update(self):
        if not self.is_active:
            return

        # 新しい位置を計算
        new_pos = self.positions[-1] + self.direction * BALL_SPEED
        
        # 壁との衝突判定
        self.check_wall_collision(new_pos)
        
        # 位置を更新
        self.positions.append(new_pos.copy())

    def check_wall_collision(self, new_pos):
        x, y = new_pos
        
        # 左右の壁との衝突
        if x < 0 or x > ROOM_WIDTH:
            self.direction[0] *= -1  # x方向を反転
            self.reflection_count += 1
            
        # 上下の壁との衝突
        if y < 0 or y > ROOM_HEIGHT:
            self.direction[1] *= -1  # y方向を反転
            self.reflection_count += 1
            
        # 反射回数チェック
        if self.reflection_count >= MAX_REFLECTIONS:
            self.is_active = False

# シミュレーション初期化
balls = [SoundBall(SOURCE_POS, np.random.randn(2)) for _ in range(NUM_BALLS)]

# アニメーション設定
fig, ax = plt.subplots()
ax.set_xlim(0, ROOM_WIDTH)
ax.set_ylim(0, ROOM_HEIGHT)
ax.set_aspect('equal')

# 部屋の描画
room_rect = plt.Rectangle((0,0), ROOM_WIDTH, ROOM_HEIGHT, fill=False, edgecolor='black')
ax.add_patch(room_rect)

# 音源の描画
source_marker, = ax.plot(SOURCE_POS[0], SOURCE_POS[1], 'ro')

# ボールの描画
scat = ax.scatter([], [], s=10, alpha=0.5)

def update(frame):
    active_balls = [ball for ball in balls if ball.is_active]
    
    # ボールの更新
    for ball in active_balls:
        ball.update()
    
    # 描画データの更新
    if active_balls:
        positions = np.array([ball.positions[-1] for ball in active_balls])
        scat.set_offsets(positions)
    else:
        scat.set_offsets([])
    
    return scat,

ani = FuncAnimation(fig, update, frames=MAX_STEPS, interval=20, blit=True)
plt.show()

# データ出力
with open('simulation_data.txt', 'w') as f:
    f.write("初期位置\t最終位置\t反射回数\n")
    for ball in balls:
        init_pos = ball.positions[0]
        final_pos = ball.positions[-1]
        f.write(f"{init_pos[0]:.2f},{init_pos[1]:.2f}\t")
        f.write(f"{final_pos[0]:.2f},{final_pos[1]:.2f}\t")
        f.write(f"{ball.reflection_count}\n") 