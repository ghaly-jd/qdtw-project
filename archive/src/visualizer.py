import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from loader import load_skeleton_file


# Kinect 20-joint edges
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Spine
    (1, 5), (5, 6), (6, 7),                 # Left Arm
    (1, 9), (9, 10), (10, 11),              # Right Arm
    (1, 12), (12, 13), (13, 14),            # Left Leg
    (1, 16), (16, 17), (17, 18),            # Right Leg
    (1, 8), (8, 15), (15, 19)               # Neck and Head
]

def visualize_skeleton_sequence(seq):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        frame = seq[frame_idx]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.5)
        ax.set_zlim(0, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Z (up)")
        ax.set_zlabel("Y")
        ax.set_title(f"Frame {frame_idx+1}/{len(seq)}")
        ax.view_init(elev=20, azim=45)  # top down tilted


        for i, j in EDGES:
            x = [frame[i][0], frame[j][0]]
            y = [frame[i][2], frame[j][2]]  # Z as up
            z = [frame[i][1], frame[j][1]]  # Y as depth
            ax.plot(x, y, z, 'bo-', linewidth=2)

    ani = FuncAnimation(fig, update, frames=len(seq), interval=100)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    seq = load_skeleton_file("msr_action_data/a01_s01_e01_skeleton.txt")
    print("seq shape:", seq.shape)
    center = np.mean(seq[:, 1], axis=0)  # use SpineBase (joint 1) as reference
    seq -= center  # center the body
    seq = seq / np.max(np.abs(seq))  # scale to [-1, 1]


    # Normalize coordinates if needed
    if np.max(seq) > 10:
        seq = seq / 1000.0

    visualize_skeleton_sequence(seq)
