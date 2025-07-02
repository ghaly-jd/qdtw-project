import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# -- Helper class
class Bunch(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

# -- Label extraction
def full_fname2_str(data_dir, fname, sep_char):
    fnametostr = ''.join(fname).replace(data_dir, '')
    ind = int(fnametostr.index(sep_char))
    label = int(fnametostr[ind + 1:ind + 3])
    return label

# -- Dataset reader
def read(data_dir):
    print('Loading MSR 3D Data, data directory %s' % data_dir)
    data, labels, lens, subjects = [], [], [], []
    documents = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir)) if d.endswith("_skeleton.txt")]
    for file in documents:
        action = np.loadtxt(file)[:, :3].flatten()
        labels.append(full_fname2_str(data_dir, file, 'a'))
        frame_size = len(action) // 60
        lens.append(frame_size)
        action = action.reshape(frame_size, 60)
        new_act = [frame for frame in action]
        data.append(new_act)
        subjects.append(full_fname2_str(data_dir, file, 's'))
    print("All files read!")
    return data, labels, lens

# -- Load single file as skeleton (for animation)
def loadData(data_dir, action, subject, instance):
    path = os.path.join(data_dir, f'a{action:02d}_s{subject:02d}_e{instance:02d}_skeleton.txt')
    raw = np.loadtxt(path)
    data = raw.reshape((raw.shape[0] // 20, 20, 4))
    return data[:, :, :3]  # remove confidence

# -- Animate single 3D skeleton sequence
def animate_3d_skeleton(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        ax.set_xlim3d([0.0, 300.0])
        ax.set_ylim3d([0.0, 1400.0])
        ax.set_zlim3d([300.0, 0.0])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f"Frame {i+1}")
        ax.scatter(data[i,:,0], data[i,:,2], data[i,:,1], c='b', s=20)

    anim = FuncAnimation(fig, animate, frames=data.shape[0], interval=100)
    plt.show()
    return anim

if __name__ == "__main__":
    MSR_data_dir = 'msr_action_data'

    for action in range(1, 21):         # 20 action classes
        for subject in range(1, 11):    # 10 subjects
            for instance in range(1, 4):  # 3 instances per subject
                try:
                    skeleton_data = loadData(MSR_data_dir, action, subject, instance)
                    print(f"Showing a{action:02d}_s{subject:02d}_e{instance:02d}")
                    animate_3d_skeleton(skeleton_data)
                except Exception as e:
                    print(f"Skipped a{action:02d}_s{subject:02d}_e{instance:02d} — {e}")
