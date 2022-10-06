import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.animation as FuncAnimation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D


def visualize_motion(motion_info, save_path2):
    mocap_kinematic_chain = [
        [0, 1, 2, 3],
        [0, 12, 13, 14, 15],
        [0, 16, 17, 18, 19],
        [1, 4, 5, 6, 7],
        [1, 8, 9, 10, 11],
  
    ]
    plot_3d_motion_v2(
        motion_info, mocap_kinematic_chain, save_path2, interval=80, dataset="mocap"
    )


def plot_3d_motion_v2(motion, kinematic_tree, save_path, interval=50, dataset="mocap"):
    matplotlib.use("Agg")

    def init(ax):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if dataset == "mocap":
            ax.set_ylim(-1, 3)
            ax.set_xlim(-3, 3)
            ax.set_zlim(-3, 3)
        else:
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    init(ax)

    data = np.array(motion, dtype=float)
    colors = ["red", "magenta", "black", "green", "blue","yellow"]
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    # print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        if dataset == "mocap":
            ax.view_init(elev=110, azim=-90)
        else:
            ax.view_init(elev=110, azim=90)
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(
                motion[index, chain, 0],
                motion[index, chain, 1],
                motion[index, chain, 2],
                linewidth=4.0,
                color=color,
            )

        if index % 5 == 0:
            ax.text(
                0,
                -2 + index * 0.05,
                index * 0.05,
                f"frame: {index}",
                color="blue",
                fontsize=10,
            )

        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.set_zticklabels([])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_number,
        interval=interval,
        repeat=False,
        repeat_delay=200,
    )

    ani.save(save_path, writer="pillow")
    plt.close()
