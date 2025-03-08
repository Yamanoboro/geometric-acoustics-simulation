'''2D Geometric Acoustics Simulation with Polygon Room
Visualize the behavior of acoustic particles in a regular polygon room,
Set parameters and run simulation, save animation to file
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import RegularPolygon
from datetime import datetime

def create_polygon_vertices(n_sides, radius, center):
    """ポリゴン頂点生成（スケーリング修正版）"""
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    vertices = np.array([
        (
            radius * np.cos(angle - np.pi/2) + center[0],  # 回転補正
            radius * np.sin(angle - np.pi/2) + center[1]
        ) for angle in angles
    ])
    return vertices

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
        sub_steps = 20
        small_step = ball_speed / sub_steps

        for _ in range(sub_steps):
            next_pos = current_pos + self.direction * small_step
            if is_point_inside_polygon(next_pos, self.vertices):
                current_pos = next_pos
            else:
                # 正確な衝突点検出
                collision_found = False
                min_t = 1.0
                collision_point = None
                collision_edge = None
                
                # 全てのエッジに対して交差判定
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
                    # 法線ベクトルの方向修正
                    edge_vec = np.array(collision_edge[1]) - np.array(collision_edge[0])
                    edge_normal = np.array([-edge_vec[1], edge_vec[0]])
                    edge_normal = edge_normal / np.linalg.norm(edge_normal)
                    
                    # 法線方向のテスト方法を改善
                    midpoint = (np.array(collision_edge[0]) + np.array(collision_edge[1])) / 2
                    test_normal = midpoint + edge_normal * 0.1
                    if not is_point_inside_polygon(test_normal, self.vertices):
                        edge_normal = -edge_normal

                    # 反射方向の計算
                    self.direction = self.direction - 2 * np.dot(self.direction, edge_normal) * edge_normal
                    self.reflection_count += 1
                    
                    # 反射後の位置調整（壁の内側に戻す）
                    current_pos = collision_point + edge_normal * 0.01  # 微小距離を追加
                    
                    if self.reflection_count >= self.max_reflections:
                        self.is_active = False
                        return

                    # デバッグ用可視化コードを削除/コメントアウト
                    # if hasattr(self, 'ax') and self.ax is not None:
                    #     self.ax.plot([collision_point[0], collision_point[0]+edge_normal[0]], 
                    #                 [collision_point[1], collision_point[1]+edge_normal[1]], 
                    #                 'r-', lw=2)
                else:
                    # 衝突点が見つからない場合の処理
                    current_pos = next_pos

        self.positions.append(current_pos.copy())

class SimulationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Geometric Acoustics Simulation (Polygon)")
        
        # Parameter input frame
        input_frame = ttk.LabelFrame(self.root, text="Simulation Parameters", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # Input fields
        ttk.Label(input_frame, text="Number of Sides:").grid(row=0, column=0, sticky="w")
        self.n_sides = ttk.Entry(input_frame)
        self.n_sides.insert(0, "6")  # Default hexagon
        self.n_sides.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Room Radius (m):").grid(row=1, column=0, sticky="w")
        self.room_radius = ttk.Entry(input_frame)
        self.room_radius.insert(0, "5.0")
        self.room_radius.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Source X (m):").grid(row=2, column=0, sticky="w")
        self.source_x = ttk.Entry(input_frame)
        self.source_x.insert(0, "0.0")
        self.source_x.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Source Y (m):").grid(row=3, column=0, sticky="w")
        self.source_y = ttk.Entry(input_frame)
        self.source_y.insert(0, "0.0")
        self.source_y.grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Number of Particles:").grid(row=4, column=0, sticky="w")
        self.num_balls = ttk.Entry(input_frame)
        self.num_balls.insert(0, "500")
        self.num_balls.grid(row=4, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Particle Size:").grid(row=5, column=0, sticky="w")
        self.particle_size = ttk.Entry(input_frame)
        self.particle_size.insert(0, "10")
        self.particle_size.grid(row=5, column=1, padx=5, pady=2)

        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)

        # Preview button
        self.preview_button = ttk.Button(button_frame, text="Preview", command=self.preview_conditions)
        self.preview_button.pack(side=tk.LEFT, padx=5)

        # Run button
        self.run_button = ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Quit button
        self.quit_button = ttk.Button(button_frame, text="Quit", command=self.quit_application)
        self.quit_button.pack(side=tk.LEFT, padx=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=5)

        # Initial display
        self.preview_conditions()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)

        self.root.mainloop()

        self.debug_mode = False  # デバッグモードフラグ追加

    def preview_conditions(self):
        """Preview the input conditions"""
        try:
            n_sides = int(self.n_sides.get())
            radius = float(self.room_radius.get())
            source_x = float(self.source_x.get())
            source_y = float(self.source_y.get())
            num_balls = int(self.num_balls.get())
            particle_size = float(self.particle_size.get())

            # Clear plot
            self.ax.clear()

            # Create polygon vertices
            center = np.array([0, 0])
            vertices = create_polygon_vertices(n_sides, radius, center)

            # Draw room
            polygon = RegularPolygon((0, 0), n_sides, radius=radius, 
                                   orientation=0, fill=False, 
                                   edgecolor='black')
            self.ax.add_patch(polygon)

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

            self.ax.legend()
            self.canvas.draw()

        except ValueError as e:
            print("Invalid input values. Please enter correct numbers.")
            return

    def run_simulation(self):
        # Get parameters
        n_sides = int(self.n_sides.get())
        radius = float(self.room_radius.get())
        source_x = float(self.source_x.get())
        source_y = float(self.source_y.get())
        num_balls = int(self.num_balls.get())
        particle_size = float(self.particle_size.get())

        # Constants
        BALL_SPEED = 0.05  # Reduced from 0.1
        MAX_STEPS = 1000   # Increased from 500 to compensate for slower speed

        # Create room geometry
        center = np.array([0, 0])
        vertices = create_polygon_vertices(n_sides, radius, center)
        SOURCE_POS = np.array([source_x, source_y])

        # Clear plot
        self.ax.clear()

        # Initialize simulation
        self.balls = [SoundBall(SOURCE_POS, np.random.randn(2), vertices, radius, self.ax) 
                     for _ in range(num_balls)]

        # Setup animation
        self.ax.set_xlim(-radius*1.2, radius*1.2)
        self.ax.set_ylim(-radius*1.2, radius*1.2)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        # Draw room
        polygon = RegularPolygon((0, 0), n_sides, radius=radius, 
                               orientation=0, fill=False, 
                               edgecolor='black')
        self.ax.add_patch(polygon)

        # Draw source
        self.ax.plot(SOURCE_POS[0], SOURCE_POS[1], 'ro')

        # Initialize particle scatter
        self.scat = self.ax.scatter([], [], s=particle_size, alpha=0.5)

        def update(frame):
            active_balls = [ball for ball in self.balls if ball.is_active]
            for ball in active_balls:
                ball.update(BALL_SPEED)
            if active_balls:
                positions = np.array([ball.positions[-1] for ball in active_balls])
                self.scat.set_offsets(positions)
            else:
                self.scat.set_offsets(np.zeros((0, 2)))
            return self.scat,

        print('Creating animation...')
        self.ani = FuncAnimation(self.fig, update, frames=MAX_STEPS, 
                               interval=20, blit=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'polygon_simulation_{timestamp}.mp4'
        print(f'Saving animation... ({filename})')
        self.ani.save(filename, writer='ffmpeg', fps=30)
        print(f'Saved to {filename}')
        
        # Save data
        data_filename = f'polygon_simulation_data_{timestamp}.txt'
        with open(data_filename, 'w') as f:
            f.write("Initial Position\tFinal Position\tReflection Count\n")
            for ball in self.balls:
                init_pos = ball.positions[0]
                final_pos = ball.positions[-1]
                f.write(f"{init_pos[0]:.2f},{init_pos[1]:.2f}\t")
                f.write(f"{final_pos[0]:.2f},{final_pos[1]:.2f}\t")
                f.write(f"{ball.reflection_count}\n")

        self.canvas.draw()

    def quit_application(self):
        """Quit the application"""
        print("Exiting program...")
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        import sys
        sys.exit(0)

if __name__ == "__main__":
    app = SimulationGUI() 