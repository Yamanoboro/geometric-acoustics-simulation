'''2D Geometric Acoustics Simulation with Polygon Room - Streamlit Version
Visualize the behavior of acoustic particles in a regular polygon room,
Set parameters and run simulation, download animation files
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st
from datetime import datetime
import os
import io
import base64
from PIL import Image
from matplotlib.patches import RegularPolygon
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    def __init__(self, start_pos, direction, vertices, radius, max_reflections):
        self.positions = [start_pos.copy()]
        self.direction = direction / np.linalg.norm(direction)
        self.reflection_count = 0
        self.is_active = True
        self.vertices = vertices
        self.edges = get_polygon_edges(vertices)
        self.radius = radius
        self.max_reflections = max_reflections
    
    def update(self, ball_speed):
        if not self.is_active:
            return

        current_pos = self.positions[-1]
        sub_steps = 20  # Reduced from 20 to improve performance
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
                    current_pos = collision_point + edge_normal * 0.015
                    
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
                if closest_vertex_dist < 0.12:  # Wider threshold for initial check
                    if closest_vertex_dist < 0.06:  # Actual collision threshold
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
                        current_pos = closest_vertex + vertex_to_ball * 0.12
                        vertex_collision = True
                
                if not vertex_collision:
                    current_pos = next_pos

        self.positions.append(current_pos.copy())

def preview_conditions():
    """条件プレビュー表示"""
    try:
        n_sides = st.session_state.n_sides
        radius = st.session_state.radius
        source_x = st.session_state.source_x
        source_y = st.session_state.source_y
        num_balls = st.session_state.num_balls
        particle_size = st.session_state.particle_size
        grid_spacing = st.session_state.grid_spacing

        # プロットの作成
        fig, ax = plt.subplots(figsize=(8, 8))

        # グリッドの追加
        limit = radius * 1.2
        
        # 0を中心にグリッド線を計算
        max_grid = int(np.ceil(limit / grid_spacing)) * grid_spacing
        x_ticks = np.arange(-max_grid, max_grid + grid_spacing, grid_spacing)
        y_ticks = np.arange(-max_grid, max_grid + grid_spacing, grid_spacing)
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, linestyle='-', alpha=0.7)
        
        # 目盛りラベルに単位を追加
        ax.set_xticklabels([f"{x:.0f}m" if x == int(x) else f"{x:.1f}m" for x in x_ticks])
        ax.set_yticklabels([f"{y:.0f}m" if y == int(y) else f"{y:.1f}m" for y in y_ticks])

        # 多角形の頂点を計算
        center = np.array([0, 0])
        vertices = create_polygon_vertices(n_sides, radius, center)
        
        # 多角形を描画（頂点を閉じるため最初の点を再追加）
        x = np.append(vertices[:,0], vertices[0,0])
        y = np.append(vertices[:,1], vertices[0,1])
        ax.plot(x, y, 'k-', lw=2)
        
        # 頂点を描画
        ax.scatter(vertices[:,0], vertices[:,1], c='red', s=50)
        
        # 頂点情報をセッションに保存
        st.session_state.room_vertices = vertices

        # 音源を描画
        ax.plot(source_x, source_y, 'ro', label='Source')

        # サンプル粒子を描画
        ax.scatter([source_x], [source_y], s=particle_size, 
                  alpha=0.5, label='Particle Sample')

        # 軸の設定
        ax.set_xlim(-radius*1.2, radius*1.2)
        ax.set_ylim(-radius*1.2, radius*1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # 条件テキストの追加
        condition_text = f"Number of Sides: {n_sides}\n"
        condition_text += f"Room Radius: {radius} m\n"
        condition_text += f"Source Position: ({source_x}, {source_y}) m\n"
        condition_text += f"Number of Particles: {num_balls}"
        
        ax.text(0.98, 0.98, condition_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 凡例を表示
        ax.legend()
        
        plt.tight_layout()
        
        return fig

    except Exception as e:
        st.error(f"プレビュー表示中にエラーが発生しました: {str(e)}")
        return None

def run_simulation():
    """シミュレーション実行"""
    try:
        # パラメータの取得
        n_sides = st.session_state.n_sides
        radius = st.session_state.radius
        source_x = st.session_state.source_x
        source_y = st.session_state.source_y
        num_balls = st.session_state.num_balls
        max_reflections = st.session_state.max_reflections
        # 重要: 最大反射回数が5以下であることを確認
        max_reflections = min(max_reflections, 5)
        particle_size = st.session_state.particle_size
        grid_spacing = st.session_state.grid_spacing

        # アニメーションの進行状況を表示するためのプレースホルダ
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 強制終了ボタン用のプレースホルダ
        stop_button_placeholder = st.empty()
        
        # 定数
        BALL_SPEED = st.session_state.ball_speed  # UIで設定した速度を使用
        MAX_STEPS = 1000  # 元のPolygonバージョンと同じ
        
        # 多角形の設定
        center = np.array([0, 0])
        SOURCE_POS = np.array([source_x, source_y])
        
        # 頂点情報を使用
        vertices = st.session_state.room_vertices
        
        # ボールの初期化
        balls = []
        angles = np.linspace(0, 2*np.pi, num_balls, endpoint=False)
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            ball = SoundBall(SOURCE_POS, direction, vertices, radius, max_reflections)
            balls.append(ball)
        
        # シミュレーション開始時の状態を表示
        print(f"Starting simulation with {num_balls} balls, max reflections: {max_reflections}")
        
        # アニメーションフレームを格納するリスト
        frames = []
        
        # プログレスバーの更新用
        frame_interval = max(1, MAX_STEPS // 100)
        
        # 初期フレームを生成
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # グリッドの追加
        limit = radius * 1.2
        max_grid = int(np.ceil(limit / grid_spacing)) * grid_spacing
        x_ticks = np.arange(-max_grid, max_grid + grid_spacing, grid_spacing)
        y_ticks = np.arange(-max_grid, max_grid + grid_spacing, grid_spacing)
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, linestyle='-', alpha=0.7)
        
        # 目盛りラベルに単位を追加
        ax.set_xticklabels([f"{x:.0f}m" if x == int(x) else f"{x:.1f}m" for x in x_ticks])
        ax.set_yticklabels([f"{y:.0f}m" if y == int(y) else f"{y:.1f}m" for y in y_ticks])
        
        # 多角形を描画
        x = np.append(vertices[:,0], vertices[0,0])
        y = np.append(vertices[:,1], vertices[0,1])
        ax.plot(x, y, 'k-', lw=2)
        
        # 音源を描画
        ax.plot(SOURCE_POS[0], SOURCE_POS[1], 'ro', label='Source')
        
        # 軸の設定
        ax.set_xlim(-radius*1.2, radius*1.2)
        ax.set_ylim(-radius*1.2, radius*1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # 凡例を表示
        ax.legend()
        
        plt.tight_layout()
        
        # 各フレームを生成
        empty_frames = 0
        
        # 強制終了フラグ
        stop_simulation = False
        
        # 強制終了ボタンを表示
        if stop_button_placeholder.button("シミュレーション強制終了", key="stop_button"):
            stop_simulation = True
            status_text.text("シミュレーションを強制終了しました。アニメーションを生成中...")
            
        for frame in range(MAX_STEPS):
            # 強制終了ボタンがクリックされたかチェック
            if stop_simulation:
                break
            
            # ボールの更新
            active_balls = [ball for ball in balls if ball.is_active]
            
            # 現在のアクティブボール数を表示（10フレームごと）
            if frame % 10 == 0:
                print(f"Frame {frame}: Active balls: {len(active_balls)}/{len(balls)}")
            
            for ball in active_balls:
                ball.update(BALL_SPEED)
            
            # スキャッタープロットの更新
            if active_balls:
                empty_frames = 0
                
                # 反射回数でグループ化して異なる色で表示
                for reflection_count in range(max_reflections + 1):
                    balls_with_reflection = [ball for ball in active_balls if ball.reflection_count == reflection_count]
                    if balls_with_reflection:
                        positions = np.array([ball.positions[-1] for ball in balls_with_reflection])
                        # 反射回数に応じた色を設定
                        colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
                        color = colors[min(reflection_count, len(colors)-1)]
                        label = f"{reflection_count}回反射" if frame == 0 else "_nolegend_"
                        ax.scatter(positions[:, 0], positions[:, 1], s=particle_size, 
                                  color=color, alpha=0.7, label=label)
            else:
                empty_frames += 1
                if empty_frames >= 100:  # 非アクティブ判定を大幅に緩和（より長く観察可能に）
                    inactive_count = len(balls) - len(active_balls)
                    inactive_percent = inactive_count / len(balls) * 100
                    status_text.text(f"シミュレーション終了：完了: {inactive_count}個 ({inactive_percent:.1f}%) | 全てのボールが非アクティブになりました")
                    break
            
            # 反射回数の統計情報を計算
            # 最大反射回数に達したボールの数（非アクティブなボールも含む）
            reached_max_reflection = [ball for ball in balls if ball.reflection_count >= max_reflections]
            # ちょうど最大反射回数に達したボールの数（非アクティブなボールも含む）
            exact_max_reflection = [ball for ball in balls if ball.reflection_count == max_reflections]
            # 最大反射回数に達したボールの位置を表示（現在アクティブかに関わらず）
            completed_positions = np.array([ball.positions[-1] for ball in reached_max_reflection]) if reached_max_reflection else np.empty((0, 2))
            if len(completed_positions) > 0:
                ax.scatter(completed_positions[:, 0], completed_positions[:, 1], s=particle_size*1.5, 
                          color='red', alpha=0.7, marker='x')  # 最大反射回数に達したボールを赤い×で表示
            avg_reflections = sum(ball.reflection_count for ball in balls) / len(balls)
            
            # 反射回数の状況をステータスに表示
            status_percent = len(reached_max_reflection) / len(balls) * 100
            inactive_count = len(balls) - len(active_balls)
            inactive_percent = inactive_count / len(balls) * 100
            status_text.text(
                f"フレーム: {frame}/{MAX_STEPS} | " 
                f"アクティブ粒子: {len(active_balls)}/{len(balls)} | "
                f"平均反射: {avg_reflections:.1f}/{max_reflections} | "
                f"完了: {inactive_count}個 ({inactive_percent:.1f}%)"
            )
            
            # フレーム数と音速換算時間をプロットに表示
            # 音速: 340m/s での時間換算 (ms)
            sound_speed = 340  # m/s
            
            # 1フレームあたりの移動距離(m)に基づいて時間を計算
            # ボールの速度(m/フレーム) ÷ 音速(m/s) = 時間(s/フレーム)
            time_per_frame_s = BALL_SPEED / sound_speed  # 1フレームあたりの時間（秒）
            total_time_ms = frame * time_per_frame_s * 1000  # 合計時間（ミリ秒）
            
            # 距離も表示（フレーム数 × ボールの速度）
            distance_m = frame * BALL_SPEED  # 移動した距離（メートル）
            
            time_text = f"Frame: {frame} | Distance: {distance_m:.2f} m | Time: {total_time_ms:.2f} ms (Sound speed: 340 m/s)"
            
            # 以前のテキストがあれば削除
            for txt in ax.texts:
                if hasattr(txt, 'counter_text'):
                    txt.remove()
            
            # 新しいテキストを追加
            text = ax.text(0.02, 0.98, time_text, transform=ax.transAxes, 
                          fontsize=14, weight='bold', verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', alpha=0.9))
            text.counter_text = True  # 識別用の属性を追加
            
            # 全ての粒子が最大反射回数に達したかチェック（非アクティブになったかで判断）
            if len(active_balls) == 0:
                status_text.text(f"シミュレーション完了！完了: {len(balls)}個 (100%) | 全ての粒子が非アクティブになりました")
                break
            
            # フレームを画像として保存
            fig.canvas.draw()
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            img = Image.fromarray(buf)
            frames.append(img)
            
            # 不要な散布図を削除（メモリ節約）
            if frame < MAX_STEPS - 1:  # 最終フレーム以外
                for coll in ax.collections:
                    if coll.get_label() != 'Source':  # 音源は残す
                        coll.remove()
            
            # プログレスバーとステータスの更新
            if frame % frame_interval == 0:
                # 完了率（％）に基づいてプログレスバーを更新
                # 1. 非アクティブになったボールの割合
                # 2. 全ボールの平均反射回数の割合
                completion_by_inactive = inactive_percent / 100.0  # 非アクティブボールの割合（0～1）
                completion_by_reflection = avg_reflections / max_reflections  # 平均反射回数の割合（0～1）
                
                # 両方の指標を組み合わせて総合的な完了率を計算
                # 非アクティブ率を優先し、残りを平均反射率で補完
                progress = completion_by_inactive + (1.0 - completion_by_inactive) * completion_by_reflection
                progress = min(1.0, progress)  # 1.0を超えないように
                progress_bar.progress(progress)
                status_text.text(f"シミュレーション実行中... 完了: {inactive_count}個 ({inactive_percent:.1f}%) | 進捗率: {progress*100:.1f}%")
                
                # 強制終了ボタンの状態を更新
                if stop_button_placeholder.button("シミュレーション強制終了", key=f"stop_button_{frame}"):
                    stop_simulation = True
                    status_text.text("シミュレーションを強制終了しました。アニメーションを生成中...")
                    break
        
        # プログレスバーを完了に設定
        progress_bar.progress(1.0)
        
        # 強制終了かどうかに応じてメッセージを変更（アニメーション生成中）
        if stop_simulation:
            status_text.text(f"シミュレーションは強制終了されました。完了: {inactive_count}個 ({inactive_percent:.1f}%) | アニメーションを生成中...")
        else:
            status_text.text(f"シミュレーション完了！完了: {len(balls)}個 (100%) | アニメーションを生成中...")
        
        # シミュレーション結果を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # GIFアニメーションの生成とダウンロードボタンの作成
        gif_bytes = io.BytesIO()
        frames[0].save(
            gif_bytes,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=25,  # milliseconds - 2倍の速さで再生
            loop=0
        )
        
        # シミュレーションデータの生成
        data_io = io.StringIO()
        data_io.write("Initial Position\tFinal Position\tReflection Count\n")
        for ball in balls:
            init_pos = ball.positions[0]
            final_pos = ball.positions[-1] if ball.positions else init_pos
            data_io.write(f"{init_pos[0]:.2f},{init_pos[1]:.2f}\t")
            data_io.write(f"{final_pos[0]:.2f},{final_pos[1]:.2f}\t")
            data_io.write(f"{ball.reflection_count}\n")
        
        # 終了メッセージの表示（アニメーション生成後）
        if stop_simulation:
            status_text.text(f"強制終了されたシミュレーションのアニメーションが生成されました。完了: {inactive_count}個 ({inactive_percent:.1f}%)")
        else:
            status_text.text("シミュレーションが完了しました！")
        
        # 最終的なグラフの表示
        st.pyplot(fig)
        
        # ダウンロードボタンのための辞書に保存
        st.session_state.downloads = {
            'gif': gif_bytes.getvalue(),
            'data': data_io.getvalue(),
            'timestamp': timestamp,
            'n_sides': n_sides
        }
        
        return True
    
    except Exception as e:
        st.error(f"シミュレーション実行中にエラーが発生しました: {str(e)}")
        # デバッグ情報の出力を削除
        # import traceback
        # traceback.print_exc()
        return False

def main():
    st.set_page_config(
        page_title="Geometric Acoustics Simulation",
        page_icon="🔊",
        layout="wide"
    )
    
    st.title("多角形の簡易幾何音響シミュレーション")
    
    # カスタムCSS
    st.markdown("""
    <style>
        .stButton > button {
            font-size: 24px !important;
            font-weight: bold !important;
            height: 3em !important;
            width: 100% !important;
            margin-bottom: 10px !important;
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 10px !important;
            border: 2px solid #2E7D32 !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s !important;
        }
        .stButton > button:hover {
            background-color: #2E7D32 !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3) !important;
            transform: translateY(-2px) !important;
        }
        /* シミュレーション実行ボタンのスタイル */
        button[data-testid="baseButton-secondary"] {
            background-color: #2196F3 !important;
            border: 2px solid #0D47A1 !important;
        }
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #0D47A1 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # セッション状態の初期化
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.n_sides = 6
        st.session_state.radius = 10.0
        st.session_state.source_x = 0.0
        st.session_state.source_y = 0.0
        st.session_state.num_balls = 300  # デフォルト粒子数を削減
        st.session_state.max_reflections = 3  # 最大反射回数を5に設定
        st.session_state.particle_size = 10.0
        st.session_state.grid_spacing = 1
        st.session_state.downloads = None
    
    # 2列レイアウトの作成
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("設定条件")
        
        # 部屋の形状設定
        st.subheader("部屋の形状")
        st.session_state.n_sides = st.number_input("多角形の辺の数", min_value=3, max_value=20, value=st.session_state.n_sides)
        st.session_state.radius = st.number_input("部屋の半径 (m)", min_value=1.0, max_value=50.0, value=st.session_state.radius)
        
        # 音源設定
        st.subheader("音源設定")
        st.session_state.source_x = st.number_input("音源位置 X (m)", min_value=-20.0, max_value=20.0, value=st.session_state.source_x)
        st.session_state.source_y = st.number_input("音源位置 Y (m)", min_value=-20.0, max_value=20.0, value=st.session_state.source_y)
        st.caption("※音源は赤い●で表示されます")
        
        # 音響粒子設定
        st.subheader("音響粒子設定")
        st.session_state.num_balls = st.number_input("粒子数", min_value=10, max_value=1000, value=st.session_state.num_balls)
        st.session_state.max_reflections = st.number_input("最大反射回数（＜5回）", min_value=1, max_value=5, value=st.session_state.max_reflections)
        st.session_state.particle_size = st.number_input("粒子サイズ", min_value=1.0, max_value=50.0, value=st.session_state.particle_size)
        
        st.caption("※粒子数や反射回数が増えると計算時間が長くなります")
        st.caption("※推奨: 高速処理なら72-144粒子")
        
        # 表示設定
        st.subheader("表示設定")
        grid_options = {1: "1m", 5: "5m"}
        grid_selection = st.radio("グリッド間隔", options=list(grid_options.keys()), format_func=lambda x: grid_options[x])
        st.session_state.grid_spacing = grid_selection
        
        # シミュレーション設定
        # st.subheader("設定条件")
        if 'ball_speed' not in st.session_state:
            st.session_state.ball_speed = 0.1
        st.session_state.ball_speed = st.slider("ボールの速度", min_value=0.01, max_value=0.2, value=st.session_state.ball_speed, step=0.01, 
                                              help="値を小さくすると計算精度が上がりますが、シミュレーション時間が長くなります")
        
        # アクションボタン
        # st.subheader("アクション")
        # st.markdown("---")
        
        # プレビューボタン用のコンテナ
        preview_container = st.container()
        with preview_container:
            # st.markdown("""
            # <div style="padding: 10px; border: 3px solid #4CAF50; border-radius: 10px; background-color: rgba(76, 175, 80, 0.1);">
            #     <h3 style="color: #4CAF50; text-align: center;">👁️ 条件プレビュー</h3>
            # </div>
            # """, unsafe_allow_html=True)
            preview_button = st.button("👁️ プレビュー表示", key="preview_button", use_container_width=True, type="primary")
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # シミュレーション実行ボタン用のコンテナ
        run_container = st.container()
        with run_container:
            st.markdown("""
            <div style="padding: 10px; border: 3px solid #2196F3; border-radius: 10px; background-color: rgba(33, 150, 243, 0.1);">
                <h3 style="color: #2196F3; text-align: center;">🚀 シミュレーションを実行</h3>
            </div>
            """, unsafe_allow_html=True)
            run_button = st.button("🚀 シミュレーション実行", key="run_button", use_container_width=True, type="primary")
        
        st.markdown("---")
    
    with col2:
        # プレビューまたはシミュレーション結果を表示
        if run_button:
            st.header("シミュレーション実行中...")
            if run_simulation():
                st.header("シミュレーション結果")
                
                # ダウンロードボタンを表示
                if st.session_state.downloads:
                    downloads = st.session_state.downloads
                    timestamp = downloads['timestamp']
                    n_sides = downloads['n_sides']
                    
                    # GIFダウンロードボタン
                    gif_filename = f"acoustic_polygon_{n_sides}sides_{timestamp}.gif"
                    st.download_button(
                        label="アニメーションをダウンロード (GIF)",
                        data=downloads['gif'],
                        file_name=gif_filename,
                        mime="image/gif"
                    )
                    
                    # データダウンロードボタン
                    data_filename = f"polygon_simulation_data_{timestamp}.txt"
                    st.download_button(
                        label="シミュレーションデータをダウンロード (TXT)",
                        data=downloads['data'],
                        file_name=data_filename,
                        mime="text/plain"
                    )
        
        elif preview_button:
            st.header("　　　条件プレビュー")
            fig = preview_conditions()
            if fig:
                st.pyplot(fig)
        
        else:
            # 初期状態または他のボタンが押されていない場合はプレビューを表示
            st.header("　　　条件プレビュー")
            fig = preview_conditions()
            if fig:
                st.pyplot(fig)

if __name__ == "__main__":
    main() 