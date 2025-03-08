'''2D Geometric Acoustics Simulation (Rectangle Room)
Visualize the behavior of acoustic particles in a rectangular room,
Set parameters and run simulation, save animation to file
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import os
import sys

# フォント設定 - 文字化け防止
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

class SoundBall:
    def __init__(self, start_pos, direction, room_width, room_height, max_reflections):
        self.positions = [start_pos.copy()]  # 軌跡の記録
        self.direction = direction / np.linalg.norm(direction)  # 進行方向（正規化）
        self.reflection_count = 0
        self.is_active = True
        self.room_width = room_width
        self.room_height = room_height
        self.max_reflections = max_reflections

    def update(self, ball_speed):
        if not self.is_active:
            return
        # 新しい位置を計算
        new_pos = self.positions[-1] + self.direction * ball_speed
        # 壁との衝突判定
        self.check_wall_collision(new_pos)
        # 位置を更新
        self.positions.append(new_pos.copy())

    def check_wall_collision(self, new_pos):
        x, y = new_pos
        # 左右の壁との衝突
        if x < 0 or x > self.room_width:
            self.direction[0] *= -1  # x方向を反転
            self.reflection_count += 1
        # 上下の壁との衝突
        if y < 0 or y > self.room_height:
            self.direction[1] *= -1  # y方向を反転
            self.reflection_count += 1
        # 反射回数チェック
        if self.reflection_count >= self.max_reflections:
            self.is_active = False

class SimulationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("幾何音響シミュレーション - 長方形部屋")
        self.root.geometry("1200x800")
        
        # Animation output folder
        self.output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "animations")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        # Create main frames
        self.input_frame = ttk.Frame(self.root, padding=10)
        self.input_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(self.root, padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup plot area - 凡例用の余白を確保するためにfigureサイズ調整
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 凡例表示用に余白を確保
        self.fig.subplots_adjust(right=0.85)
        
        # Add grid to the plot with default 1m spacing
        self.grid_spacing = tk.IntVar(value=1)  # Default 1m grid
        
        # Setup simulation parameters
        self.setup_inputs()
        
        # Draw initial rectangle
        self.preview_conditions()
        
        # 設定保存
        self.animation_in_progress = False
        self.ani = None
        
        # ウィンドウ閉じる処理を設定
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)

    def setup_inputs(self):
        # Title
        ttk.Label(self.input_frame, text="シミュレーション設定", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Room parameters - グループ化
        ttk.Label(self.input_frame, text="◆ 部屋の形状", font=("Arial", 10, "bold")).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(self.input_frame, text="部屋の幅 (m):").grid(row=2, column=0, sticky="w", pady=2)
        self.room_width = ttk.Entry(self.input_frame, width=10)
        self.room_width.insert(0, "10.0")
        self.room_width.grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(self.input_frame, text="部屋の高さ (m):").grid(row=3, column=0, sticky="w", pady=2)
        self.room_height = ttk.Entry(self.input_frame, width=10)
        self.room_height.insert(0, "5.0")
        self.room_height.grid(row=3, column=1, sticky="w", pady=2)
        
        # Sound source parameters
        ttk.Label(self.input_frame, text="◆ 音源設定", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(self.input_frame, text="音源位置 X (m):").grid(row=5, column=0, sticky="w", pady=2)
        self.source_x = ttk.Entry(self.input_frame, width=10)
        self.source_x.insert(0, "5.0")
        self.source_x.grid(row=5, column=1, sticky="w", pady=2)
        
        ttk.Label(self.input_frame, text="音源位置 Y (m):").grid(row=6, column=0, sticky="w", pady=2)
        self.source_y = ttk.Entry(self.input_frame, width=10)
        self.source_y.insert(0, "2.5")
        self.source_y.grid(row=6, column=1, sticky="w", pady=2)
        
        # 音源の説明
        source_note = ttk.Label(self.input_frame, text="※音源は赤い●で表示されます", 
                          font=("Arial", 8), foreground="gray")
        source_note.grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        # Simulation parameters
        ttk.Label(self.input_frame, text="◆ 音響粒子設定", font=("Arial", 10, "bold")).grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(self.input_frame, text="粒子数:").grid(row=9, column=0, sticky="w", pady=2)
        self.num_balls = ttk.Entry(self.input_frame, width=10)
        self.num_balls.insert(0, "500")
        self.num_balls.grid(row=9, column=1, sticky="w", pady=2)
        
        # Add note about calculation time
        ttk.Label(self.input_frame, text="※粒子数や反射回数が増えると計算時間が長くなります", 
                 font=("Arial", 8), foreground="red").grid(row=10, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(self.input_frame, text="※推奨: 高速処理なら72-144粒子", 
                 font=("Arial", 8), foreground="blue").grid(row=11, column=0, columnspan=2, sticky="w", pady=2)
        
        ttk.Label(self.input_frame, text="最大反射回数:").grid(row=12, column=0, sticky="w", pady=2)
        self.max_reflections = ttk.Entry(self.input_frame, width=10)
        self.max_reflections.insert(0, "10")
        self.max_reflections.grid(row=12, column=1, sticky="w", pady=2)
        
        ttk.Label(self.input_frame, text="粒子サイズ:").grid(row=13, column=0, sticky="w", pady=2)
        self.particle_size = ttk.Entry(self.input_frame, width=10)
        self.particle_size.insert(0, "10")
        self.particle_size.grid(row=13, column=1, sticky="w", pady=2)
        
        # Grid options
        ttk.Label(self.input_frame, text="◆ 表示設定", font=("Arial", 10, "bold")).grid(row=14, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(self.input_frame, text="グリッド間隔:").grid(row=15, column=0, sticky="w", pady=2)
        
        grid_frame = ttk.Frame(self.input_frame)
        grid_frame.grid(row=15, column=1, sticky="w", pady=2)
        
        ttk.Radiobutton(grid_frame, text="1m", variable=self.grid_spacing, value=1, 
                       command=self.update_grid).pack(side=tk.LEFT)
        ttk.Radiobutton(grid_frame, text="5m", variable=self.grid_spacing, value=5, 
                       command=self.update_grid).pack(side=tk.LEFT)
        
        # Output folder selection - Improved UI
        ttk.Separator(self.input_frame, orient='horizontal').grid(row=16, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(self.input_frame, text="◆ 保存設定", font=("Arial", 10, "bold")).grid(row=17, column=0, columnspan=2, sticky="w", pady=5)
        
        ttk.Label(self.input_frame, text="現在の格納フォルダー:").grid(row=18, column=0, sticky="w", pady=2)
        
        folder_frame = ttk.Frame(self.input_frame)
        folder_frame.grid(row=18, column=1, sticky="w", pady=2)
        
        # Current folder display
        self.folder_label = ttk.Label(folder_frame, text="...animations", width=15, 
                                    background="white", borderwidth=1, relief="solid")
        self.folder_label.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(folder_frame, text="フォルダ選択", command=self.select_output_folder).pack(side=tk.LEFT)
        
        # 説明文を追加
        ttk.Label(self.input_frame, text="※ここに動画ファイルが保存されます", 
                font=("Arial", 8), foreground="gray").grid(row=19, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        # Buttons
        ttk.Separator(self.input_frame, orient='horizontal').grid(row=20, column=0, columnspan=2, sticky="ew", pady=10)
        
        button_frame = ttk.Frame(self.input_frame)
        button_frame.grid(row=21, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="プレビュー", command=self.preview_conditions).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="シミュレーション実行", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="終了", command=self.quit_application).pack(side=tk.LEFT, padx=5)

    def select_output_folder(self):
        """出力フォルダを選択するダイアログを表示"""
        folder = filedialog.askdirectory(
            initialdir=self.output_folder,
            title="アニメーション保存先フォルダの選択"
        )
        if folder:  # User didn't cancel
            self.output_folder = folder
            # Update label with shortened path
            if len(folder) > 15:
                display_path = "..." + folder[-15:]
            else:
                display_path = folder
            self.folder_label.config(text=display_path)
    
    def update_grid(self):
        """グリッド間隔が変更されたときの処理"""
        self.preview_conditions()

    def preview_conditions(self):
        """プレビュー表示（部屋と音源）"""
        try:
            # パラメータの取得
            room_width = float(self.room_width.get())
            room_height = float(self.room_height.get())
            source_x = float(self.source_x.get())
            source_y = float(self.source_y.get())
            num_balls = int(self.num_balls.get())
            max_reflections = int(self.max_reflections.get())
            particle_size = float(self.particle_size.get())

            # 描画のクリア
            self.ax.clear()

            # グリッド追加
            grid_spacing = self.grid_spacing.get()
            # X軸グリッド線（0から幅まで、指定間隔）
            x_ticks = np.arange(0, room_width + grid_spacing, grid_spacing)
            # Y軸グリッド線（0から高さまで、指定間隔）
            y_ticks = np.arange(0, room_height + grid_spacing, grid_spacing)
            
            self.ax.set_xticks(x_ticks)
            self.ax.set_yticks(y_ticks)
            self.ax.grid(True, linestyle='-', alpha=0.7)
            
            # 軸ラベルに単位を追加
            self.ax.set_xticklabels([f"{x:.0f}m" if x == int(x) else f"{x:.1f}m" for x in x_ticks])
            self.ax.set_yticklabels([f"{y:.0f}m" if y == int(y) else f"{y:.1f}m" for y in y_ticks])

            # 部屋の描画
            room_rect = plt.Rectangle((0, 0), room_width, room_height, fill=False, edgecolor='black')
            self.ax.add_patch(room_rect)

            # 音源の描画
            self.ax.plot(source_x, source_y, 'ro', label='Source')

            # サンプル粒子の描画
            sample_positions = np.array([[source_x, source_y]])
            self.ax.scatter(sample_positions[:, 0], sample_positions[:, 1], 
                          s=particle_size, alpha=0.5, label='Particle Sample')

            # 軸の設定
            self.ax.set_xlim(0, room_width)
            self.ax.set_ylim(0, room_height)
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')

            # 条件テキストの追加
            condition_text = f"Room Size: {room_width}×{room_height} m\n"
            condition_text += f"Source Position: ({source_x}, {source_y}) m\n"
            condition_text += f"Number of Particles: {num_balls}\n"
            condition_text += f"Max Reflections: {max_reflections}"
            
            # テキストを配置（右上）
            self.ax.text(0.98, 0.98, condition_text,
                        transform=self.ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 凡例をグラフの外側に配置
            self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            
            # グラフレイアウトの調整（凡例の表示スペース確保）
            plt.tight_layout()
            self.fig.subplots_adjust(right=0.85)  # 右側に余白を確保
            
            self.canvas.draw()

        except ValueError as e:
            messagebox.showerror("入力エラー", "数値の入力が正しくありません")
            return

    def run_simulation(self):
        """シミュレーション実行"""
        if self.animation_in_progress:
            messagebox.showinfo("情報", "アニメーションが既に実行中です")
            return
            
        try:
            # パラメータの取得
            room_width = float(self.room_width.get())
            room_height = float(self.room_height.get())
            source_x = float(self.source_x.get())
            source_y = float(self.source_y.get())
            num_balls = int(self.num_balls.get())
            max_reflections = int(self.max_reflections.get())
            particle_size = float(self.particle_size.get())
        except ValueError:
            messagebox.showerror("入力エラー", "数値の入力が正しくありません")
            return

        # 定数パラメータ
        BALL_SPEED = 0.05  # 速度を遅く
        MAX_STEPS = 1000   # ステップ数を増加

        # 音源位置
        SOURCE_POS = np.array([source_x, source_y])

        # 描画のクリア
        self.ax.clear()
        
        # グリッド追加
        grid_spacing = self.grid_spacing.get()
        # X軸グリッド線（0から幅まで、指定間隔）
        x_ticks = np.arange(0, room_width + grid_spacing, grid_spacing)
        # Y軸グリッド線（0から高さまで、指定間隔）
        y_ticks = np.arange(0, room_height + grid_spacing, grid_spacing)
        
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        self.ax.grid(True, linestyle='-', alpha=0.7)
        
        # 軸ラベルに単位を追加
        self.ax.set_xticklabels([f"{x:.0f}m" if x == int(x) else f"{x:.1f}m" for x in x_ticks])
        self.ax.set_yticklabels([f"{y:.0f}m" if y == int(y) else f"{y:.1f}m" for y in y_ticks])
        
        # シミュレーション初期化
        self.balls = []
        angles = np.linspace(0, 2*np.pi, num_balls, endpoint=False)
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            ball = SoundBall(SOURCE_POS, direction, room_width, room_height, max_reflections)
            self.balls.append(ball)

        # アニメーション設定
        self.ax.set_xlim(0, room_width)
        self.ax.set_ylim(0, room_height)
        self.ax.set_aspect('equal')
        
        # 軸ラベルの設定
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        # 部屋の描画
        room_rect = plt.Rectangle((0, 0), room_width, room_height, fill=False, edgecolor='black')
        self.ax.add_patch(room_rect)

        # 音源の描画
        self.ax.plot(SOURCE_POS[0], SOURCE_POS[1], 'ro', label='Source')

        # ボールの描画（粒子の大きさを反映）
        self.scat = self.ax.scatter([], [], s=particle_size, alpha=0.5, label='Particles')
        
        # 凡例をグラフの外側に配置
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        # グラフレイアウトの調整（凡例の表示スペース確保）
        plt.tight_layout()
        self.fig.subplots_adjust(right=0.85)  # 右側に余白を確保
        
        # Initialize empty frame counter to count frames with no active balls
        empty_frames = 0

        def update(frame):
            nonlocal empty_frames
            active_balls = [ball for ball in self.balls if ball.is_active]
            # Update active balls
            for ball in active_balls:
                ball.update(BALL_SPEED)
            if active_balls:
                empty_frames = 0
                positions = np.array([ball.positions[-1] for ball in active_balls])
                self.scat.set_offsets(positions)
            else:
                empty_frames += 1
                self.scat.set_offsets(np.zeros((0, 2)))
                if empty_frames >= 50:  # 1 second delay at 20ms interval
                    print('All balls inactive. Stopping simulation after 1 second delay.')
                    self.ani.event_source.stop()
                    self.animation_in_progress = False
                    self.save_animation_results()
            return self.scat,

        print('Creating animation...')
        self.animation_in_progress = True
        self.ani = FuncAnimation(self.fig, update, frames=MAX_STEPS, interval=20, blit=True)
        
        self.canvas.draw()
        
    def save_animation_results(self):
        """アニメーション結果の保存（別メソッドに分離）"""
        if messagebox.askyesno("保存確認", "アニメーションを保存しますか？"):
            try:
                # Create animation filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                animation_filename = os.path.join(self.output_folder, f"rectangle_animation_{timestamp}.mp4")
                print(f'Saving animation... ({animation_filename})')
                
                self.root.title("アニメーション保存中... お待ちください")
                self.root.update()
                
                self.ani.save(animation_filename, writer='ffmpeg', fps=30)
                print(f'Saved to {animation_filename}')
                
                self.root.title("幾何音響シミュレーション - 長方形部屋")
                
                # Save data
                data_filename = os.path.join(self.output_folder, f'rectangle_data_{timestamp}.txt')
                with open(data_filename, 'w') as f:
                    f.write("Initial Position\tFinal Position\tReflection Count\n")
                    for ball in self.balls:
                        init_pos = ball.positions[0]
                        final_pos = ball.positions[-1]
                        f.write(f"{init_pos[0]:.2f},{init_pos[1]:.2f}\t")
                        f.write(f"{final_pos[0]:.2f},{final_pos[1]:.2f}\t")
                        f.write(f"{ball.reflection_count}\n")
                
                messagebox.showinfo("保存完了", f"アニメーションと解析データが保存されました:\n{animation_filename}")
            except Exception as e:
                messagebox.showerror("保存エラー", f"アニメーション保存中にエラーが発生しました: {str(e)}")

    def quit_application(self):
        """アプリケーションの終了"""
        if self.animation_in_progress:
            if not messagebox.askyesno("確認", "シミュレーション実行中です。終了しますか？"):
                return
        
        print("Closing application...")
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        
# メインプログラム
if __name__ == "__main__":
    try:
        app = SimulationGUI()
        app.root.mainloop()
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 