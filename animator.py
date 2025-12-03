import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

def create_segments(nodes, edges):
    """
    Converts nodes (2,N) and edges (2,M) into a list of segments
    format required for LineCollection: (M, 2, 2)
    """

    start_indices = edges[0, :]
    end_indices = edges[1, :]
    
    start_coords = nodes[:, start_indices].T # (M, 2)
    end_coords = nodes[:, end_indices].T     # (M, 2)
    
    segments = np.stack((start_coords, end_coords), axis=1)
    return segments

def save_animation(history_data, static_die_nodes, sheet_edges, punch_edges, die_edges, coord_min, coord_max, filename="v_bending.mp4"):
    print("Generating animation...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(coord_min[0], coord_max[0])
    ax.set_ylim(coord_min[1], coord_max[1])
    ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_title("V-Bending Simulation")

    # 1. STATIC OBJECTS (Draw once)
    # Pre-calculate die segments
    seg_die = create_segments(static_die_nodes, die_edges)
    die_lines = LineCollection(seg_die, colors='brown', linewidths=2, label='Die (Static)')
    ax.add_collection(die_lines)

    # 2. DYNAMIC OBJECTS (Initialize empty)
    sheet_lines = LineCollection([], colors='orange', linewidths=1.5, label='Sheet')
    punch_lines = LineCollection([], colors='brown', linewidths=2, label='Punch')
    
    ax.add_collection(sheet_lines)
    ax.add_collection(punch_lines)
    ax.legend(loc='upper right')

    def update(frame_idx):
        # Get data for this time step
        frame_data = history_data[frame_idx]
        current_sheet = frame_data['sheet']
        current_punch = frame_data['punch']
        time_val = frame_data['time']

        # Update Geometry
        seg_sheet = create_segments(current_sheet, sheet_edges)
        seg_punch = create_segments(current_punch, punch_edges)

        # Update Plot Objects
        sheet_lines.set_segments(seg_sheet)
        punch_lines.set_segments(seg_punch)

        ax.set_title(f"V-Bending Simulation | Time: {time_val * 1e3:.3f} ms")
        return sheet_lines, punch_lines, die_lines

    ani = FuncAnimation(fig, update, frames=len(history_data), blit=True, interval=50)
    
    try:
        ani.save(filename, fps=20, dpi=200, writer='pillow')
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Could not save video. Showing instead.\nError: {e}")
        plt.show()
