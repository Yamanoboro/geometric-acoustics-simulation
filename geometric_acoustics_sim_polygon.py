'''2D Geometric Acoustics Simulation with Polygon Room
Visualize the behavior of acoustic particles in a regular polygon room,
Set parameters and run simulation, save animation to file
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import RegularPolygon
from datetime import datetime
import os
import sys

# フォント設定 - 文字化け防止
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def create_polygon_vertices(n_sides, radius, center):
    """頂点生成（描画と計算で同一のものを使用）"""
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    vertices = []
    for angle in angles:
        x = radius * np.cos(angle) + center[0]
        y = radius * np.sin(angle) + center[1]
        vertices.append([x, y])
    return np.array(vertices)

def get_polygon_edges(vertices):
    """Get edges (line segments) of the polygon"""
    edges = []
    n = len(vertices)
    for i in range(n):
        edges.append((vertices[i], vertices[(i+1)%n]))
    return edges

def distance_point_to_line(point, line_start, line_end):
    """Calculate distance from point to line segment and projection point"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_length = np.linalg.norm(line_vec)
    line_unit_vec = line_vec / line_length
    projection_length = np.dot(point_vec, line_unit_vec)
    
    if projection_length < 0:
        return np.linalg.norm(point_vec), line_start
    elif projection_length > line_length:
        return np.linalg.norm(point - line_end), line_end
    else:
        projection_point = line_start + line_unit_vec * projection_length
        return np.linalg.norm(point - projection_point), projection_point

def is_point_inside_polygon(point, vertices):
    """Check if a point is inside the polygon using ray casting algorithm"""
    x, y = point
    n = len(vertices)
    inside = False
    
    j = n - 1
    for i in range(n):
        if ((vertices[i][1] > y) != (vertices[j][1] > y) and
            x < (vertices[j][0] - vertices[i][0]) * (y - vertices[i][1]) /
                (vertices[j][1] - vertices[i][1]) + vertices[i][0]):
            inside = not inside
        j = i
    
    return inside

def line_segment_intersection(p1, p2, p3, p4):
    """Return the intersection point and parameter t for the intersection of line segments p1->p2 and p3->p4, or None if no intersection."""
    r = p2 - p1
    s = p4 - p3
    r_cross_s = r[0]*s[1] - r[1]*s[0]
    if np.abs(r_cross_s) < 1e-8:
        return None
    t = ((p3[0] - p1[0])*s[1] - (p3[1] - p1[1])*s[0]) / r_cross_s
    u = ((p3[0] - p1[0])*r[1] - (p3[1] - p1[1])*r[0]) / r_cross_s
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t*r, t
    return None

class SoundBall:
    def __init__(self, start_pos, direction, vertices, radius, ax):
        self.positions = [start_pos.copy()]
        self.direction = direction / np.linalg.norm(direction)
        self.reflection_count = 0
        self.is_active = True
        self.vertices = vertices
        self.edges = get_polygon_edges(vertices)
        self.radius = radius
        self.max_reflections = 5
        self.ax = ax

    def update(self, ball_speed):
        if not self.is_active:
            return

        current_pos = self.positions[-1]
        sub_steps = 10  # Reduced from 20 to improve performance
        small_step = ball_speed / sub_steps

        for _ in range(sub_steps):
            next_pos = current_pos + self.direction * small_step
            
            # First check edge collisions - more common case
            collision_found = False
            
            # Check if next position would be outside
            if not is_point_inside_polygon(next_pos, self.vertices):
                min_t = 1.0
                collision_point = None
                collision_edge = None
                
                # Check all edges for intersection
                for edge in self.edges:
                    result = line_segment_intersection(
                        current_pos, next_pos, 
                        np.array(edge[0]), np.array(edge[1])
                    )
                    if result is not None:
                        pt, t_val = result
                        if t_val < min_t:
                            min_t = t_val
                            collision_point = pt
                            collision_edge = edge
                            collision_found = True

                if collision_found and collision_point is not None:
                    # Calculate normal vector
                    edge_vec = np.array(collision_edge[1]) - np.array(collision_edge[0])
                    edge_normal = np.array([edge_vec[1], -edge_vec[0]])
                    edge_normal = edge_normal / np.linalg.norm(edge_normal)
                    
                    # Make sure normal points inward
                    midpoint = (np.array(collision_edge[0]) + np.array(collision_edge[1])) / 2
                    center_to_mid = midpoint - np.mean(self.vertices, axis=0)
                    if np.dot(edge_normal, center_to_mid) < 0:
                        edge_normal = -edge_normal

                    # Calculate reflection
                    self.direction = self.direction - 2 * np.dot(self.direction, edge_normal) * edge_normal
                    self.reflection_count += 1
                    
                    # Move to reflected position
                    current_pos = collision_point + edge_normal * 0.01
                    
                    if self.reflection_count >= self.max_reflections:
                        self.is_active = False
                        return
            
            # Only check vertex collisions if no edge collision was found
            if not collision_found:
                # Optimize vertex check - only check if we're close to any vertex
                vertex_collision = False
                
                # Quick bounding check first
                closest_vertex_dist = float('inf')
                closest_vertex = None
                
                for vertex in self.vertices:
                    dist = np.linalg.norm(next_pos - vertex)
                    if dist < closest_vertex_dist:
                        closest_vertex_dist = dist
                        closest_vertex = vertex
                
                # Only do detailed check if we're close to a vertex
                if closest_vertex_dist < 0.1:  # Wider threshold for initial check
                    if closest_vertex_dist < 0.05:  # Actual collision threshold
                        # Calculate reflection direction from vertex
                        vertex_to_ball = current_pos - closest_vertex
                        vertex_to_ball = vertex_to_ball / np.linalg.norm(vertex_to_ball)
                        
                        # Reflect direction vector
                        self.direction = 2 * vertex_to_ball * np.dot(self.direction, vertex_to_ball) - self.direction
                        self.reflection_count += 1
                        
                        if self.reflection_count >= self.max_reflections:
                            self.is_active = False
                            return
                            
                        # Move away from vertex
                        current_pos = closest_vertex + vertex_to_ball * 0.1
                        vertex_collision = True
                
                if not vertex_collision:
                    current_pos = next_pos

        self.positions.append(current_pos.copy())

class SimulationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Geometric Acoustics Simulation - Polygon")
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
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add grid to the plot with default 1m spacing
        self.grid_spacing = tk.IntVar(value=1)  # Default 1m grid
        
        # Setup simulation parameters
        self.setup_inputs()
        
        # Draw initial polygon
        self.preview_conditions()
        
        # 設定保存
        self.animation_in_progress = False
        self.ani = None
        
        # ウィンドウ閉じる処理を設定
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)

    def setup_inputs(self):
        # Title
        ttk.Label(self.input_frame, text="シミュレーション設定", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Polygon parameters - グループ化
        ttk.Label(self.input_frame, text="◆ 部屋の形状", font=("Arial", 10, "bold")).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(self.input_frame, text="多角形の辺の数:").grid(row=2, column=0, sticky="w", pady=2)
        self.sides_entry = ttk.Entry(self.input_frame, width=10)
        self.sides_entry.insert(0, "6")
        self.sides_entry.grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(self.input_frame, text="部屋の半径 (m):").grid(row=3, column=0, sticky="w", pady=2)
        self.radius_entry = ttk.Entry(self.input_frame, width=10)
        self.radius_entry.insert(0, "10")
        self.radius_entry.grid(row=3, column=1, sticky="w", pady=2)
        
        # Sound source parameters
        ttk.Label(self.input_frame, text="◆ 音源設定", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(self.input_frame, text="音源位置 X (m):").grid(row=5, column=0, sticky="w", pady=2)
        self.source_x_entry = ttk.Entry(self.input_frame, width=10)
        self.source_x_entry.insert(0, "0")
        self.source_x_entry.grid(row=5, column=1, sticky="w", pady=2)
        
        ttk.Label(self.input_frame, text="音源位置 Y (m):").grid(row=6, column=0, sticky="w", pady=2)
        self.source_y_entry = ttk.Entry(self.input_frame, width=10)
        self.source_y_entry.insert(0, "0")
        self.source_y_entry.grid(row=6, column=1, sticky="w", pady=2)
        
        # 音源の説明
        source_note = ttk.Label(self.input_frame, text="※音源は赤い●で表示されます", 
                           font=("Arial", 8), foreground="gray")
        source_note.grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        # Simulation parameters
        ttk.Label(self.input_frame, text="◆ 音響粒子設定", font=("Arial", 10, "bold")).grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(self.input_frame, text="粒子数:").grid(row=9, column=0, sticky="w", pady=2)
        self.num_balls_entry = ttk.Entry(self.input_frame, width=10)
        self.num_balls_entry.insert(0, "500")
        self.num_balls_entry.grid(row=9, column=1, sticky="w", pady=2)
        
        # Add note about calculation time
        ttk.Label(self.input_frame, text="※粒子数や反射回数が増えると計算時間が長くなります", 
                  font=("Arial", 8), foreground="red").grid(row=10, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(self.input_frame, text="※推奨: 高速処理なら72-144粒子", 
                  font=("Arial", 8), foreground="blue").grid(row=11, column=0, columnspan=2, sticky="w", pady=2)
        
        ttk.Label(self.input_frame, text="最大反射回数:").grid(row=12, column=0, sticky="w", pady=2)
        self.max_reflect_entry = ttk.Entry(self.input_frame, width=10)
        self.max_reflect_entry.insert(0, "10")
        self.max_reflect_entry.grid(row=12, column=1, sticky="w", pady=2)
        
        ttk.Label(self.input_frame, text="粒子サイズ:").grid(row=13, column=0, sticky="w", pady=2)
        self.particle_size_entry = ttk.Entry(self.input_frame, width=10)
        self.particle_size_entry.insert(0, "10")
        self.particle_size_entry.grid(row=13, column=1, sticky="w", pady=2)
        
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
        folder = filedialog.askdirectory(
            initialdir=self.output_folder,
            title="Select Output Folder for Animations"
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
        self.preview_conditions()

    def preview_conditions(self):
        """プレビュー表示（ポリゴン描画修正）"""
        try:
            n_sides = int(self.sides_entry.get())
            radius = float(self.radius_entry.get())
            source_x = float(self.source_x_entry.get())
            source_y = float(self.source_y_entry.get())
            num_balls = int(self.num_balls_entry.get())
            particle_size = float(self.particle_size_entry.get())

            # Clear plot
            self.ax.clear()

            # Add grid to the preview, centered at 0
            grid_spacing = self.grid_spacing.get()
            limit = radius * 1.2
            
            # Calculate grid lines to ensure 0 is included and centered
            max_grid = int(np.ceil(limit / grid_spacing)) * grid_spacing
            x_ticks = np.arange(-max_grid, max_grid + grid_spacing, grid_spacing)
            y_ticks = np.arange(-max_grid, max_grid + grid_spacing, grid_spacing)
            
            self.ax.set_xticks(x_ticks)
            self.ax.set_yticks(y_ticks)
            self.ax.grid(True, linestyle='-', alpha=0.7)
            
            # Add tick labels with units
            self.ax.set_xticklabels([f"{x:.0f}m" if x == int(x) else f"{x:.1f}m" for x in x_ticks])
            self.ax.set_yticklabels([f"{y:.0f}m" if y == int(y) else f"{y:.1f}m" for y in y_ticks])

            # RegularPolygonを使わずに直接頂点から描画
            center = np.array([0, 0])
            vertices = create_polygon_vertices(n_sides, radius, center)
            
            # 頂点を閉じるために最初の点を再追加
            x = np.append(vertices[:,0], vertices[0,0])
            y = np.append(vertices[:,1], vertices[0,1])
            self.ax.plot(x, y, 'k-', lw=2)
            
            # デバッグ用に頂点を描画
            self.ax.scatter(vertices[:,0], vertices[:,1], c='red', s=50)
            
            # 計算用の頂点を保存（後でボール初期化時に使用）
            self.room_vertices = vertices

            # Draw source
            self.ax.plot(source_x, source_y, 'ro', label='Source')

            # Draw sample particle
            self.ax.scatter([source_x], [source_y], s=particle_size, 
                          alpha=0.5, label='Particle Sample')

            # Set axis
            self.ax.set_xlim(-radius*1.2, radius*1.2)
            self.ax.set_ylim(-radius*1.2, radius*1.2)
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')

            # Add condition text
            condition_text = f"Number of Sides: {n_sides}\n"
            condition_text += f"Room Radius: {radius} m\n"
            condition_text += f"Source Position: ({source_x}, {source_y}) m\n"
            condition_text += f"Number of Particles: {num_balls}"
            
            self.ax.text(0.98, 0.98, condition_text,
                        transform=self.ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 凡例をグラフの外側に配置
            self.ax.legend()
            
            plt.tight_layout()

            self.canvas.draw()

        except ValueError as e:
            print("Invalid input values. Please enter correct numbers.")
            return

    def run_simulation(self):
        """シミュレーション実行（プレビューと実行を分離）"""
        if self.animation_in_progress:
            messagebox.showinfo("情報", "アニメーションが既に実行中です")
            return
            
        try:
            # Get parameters
            n_sides = int(self.sides_entry.get())
            radius = float(self.radius_entry.get())
            source_x = float(self.source_x_entry.get())
            source_y = float(self.source_y_entry.get())
            num_balls = int(self.num_balls_entry.get())
            max_reflections = int(self.max_reflect_entry.get())
            particle_size = float(self.particle_size_entry.get())
        except ValueError:
            messagebox.showerror("入力エラー", "数値の入力が正しくありません")
            return

        # Constants
        BALL_SPEED = 0.05  # Reduced from 0.1
        MAX_STEPS = 1000   # Increased from 500 to compensate for slower speed

        # Create room geometry
        center = np.array([0, 0])
        vertices = create_polygon_vertices(n_sides, radius, center)
        SOURCE_POS = np.array([source_x, source_y])

        # Clear plot
        self.ax.clear()

        # Add grid to the animation view, centered at 0
        grid_spacing = self.grid_spacing.get()
        limit = radius * 1.2
        
        # Calculate grid lines to ensure 0 is included and centered
        max_grid = int(np.ceil(limit / grid_spacing)) * grid_spacing
        x_ticks = np.arange(-max_grid, max_grid + grid_spacing, grid_spacing)
        y_ticks = np.arange(-max_grid, max_grid + grid_spacing, grid_spacing)
        
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        self.ax.grid(True, linestyle='-', alpha=0.7)
        
        # Add tick labels with units
        self.ax.set_xticklabels([f"{x:.0f}m" if x == int(x) else f"{x:.1f}m" for x in x_ticks])
        self.ax.set_yticklabels([f"{y:.0f}m" if y == int(y) else f"{y:.1f}m" for y in y_ticks])

        # preview_conditionsで計算した頂点を使用
        vertices = self.room_vertices
        
        # 頂点を閉じるために最初の点を再追加
        x = np.append(vertices[:,0], vertices[0,0])
        y = np.append(vertices[:,1], vertices[0,1])
        self.ax.plot(x, y, 'k-', lw=2)

        # Initialize simulation
        self.balls = []
        angles = np.linspace(0, 2*np.pi, num_balls, endpoint=False)
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            ball = SoundBall(SOURCE_POS, direction, vertices, radius, self.ax)
            ball.max_reflections = max_reflections  # 反射回数を設定
            self.balls.append(ball)

        # Setup animation
        self.ax.set_xlim(-radius*1.2, radius*1.2)
        self.ax.set_ylim(-radius*1.2, radius*1.2)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        # Draw source
        self.ax.plot(SOURCE_POS[0], SOURCE_POS[1], 'ro', label='Source')

        # Initialize particle scatter
        self.scat = self.ax.scatter([], [], s=particle_size, alpha=0.5, label='Particles')
        
        # 凡例をグラフの外側に配置
        self.ax.legend()
        
        # Initialize empty frame counter
        empty_frames = 0

        def update(frame):
            nonlocal empty_frames
            active_balls = [ball for ball in self.balls if ball.is_active]
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
                    self.save_animation_results(n_sides)
            return [self.scat]  # Return as list to ensure proper artist handling

        print('Creating animation...')
        self.animation_in_progress = True
        self.ani = FuncAnimation(self.fig, update, frames=MAX_STEPS, 
                               interval=20, blit=True)
        
        self.canvas.draw()
    
    def save_animation_results(self, n_sides):
        """アニメーション結果の保存（別メソッドに分離）"""
        if messagebox.askyesno("保存確認", "アニメーションを保存しますか？"):
            try:
                # Create animation filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                animation_filename = os.path.join(self.output_folder, f"acoustic_polygon_{n_sides}sides_{timestamp}.mp4")
                print(f'Saving animation... ({animation_filename})')
                
                self.root.title("Saving Animation... Please Wait")
                self.root.update()
                
                self.ani.save(animation_filename, writer='ffmpeg', fps=30)
                print(f'Saved to {animation_filename}')
                
                self.root.title("Geometric Acoustics Simulation - Polygon")
                
                # Save data
                data_filename = os.path.join(self.output_folder, f'polygon_simulation_data_{timestamp}.txt')
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