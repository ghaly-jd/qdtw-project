import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def fig_shared_subspace_plane(path="fig_shared_subspace_plane.png", seed=7):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3,2))
    U_ref, _ = np.linalg.qr(A)  # 2D plane in R^3

    # rotate basis within the same plane to represent "VQD vs PCA"
    theta = np.deg2rad(32)
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
    U_pca = U_ref
    U_vqd = U_ref @ R2

    # plane wireframe
    g = np.linspace(-1.5, 1.5, 10)
    Gx, Gy = np.meshgrid(g, g)
    plane_pts = (U_ref @ np.vstack([Gx.ravel(), Gy.ravel()])).reshape(3, -1)
    PX = plane_pts[0].reshape(Gx.shape)
    PY = plane_pts[1].reshape(Gx.shape)
    PZ = plane_pts[2].reshape(Gx.shape)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(PX, PY, PZ, rstride=1, cstride=1, linewidth=0.6)

    origin = np.zeros(3); scale = 1.2
    # PCA basis (solid)
    for j in range(2):
        v = U_pca[:, j] * scale
        ax.quiver(*origin, *v, arrow_length_ratio=0.1)
    # VQD basis (dashed)
    for j in range(2):
        v = U_vqd[:, j] * scale
        ax.quiver(*origin, *v, arrow_length_ratio=0.1, linestyle='dashed')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Shared 2D Subspace in $\\mathbb{R}^3$: PCA (solid) vs VQD (dashed)')
    lims = np.array([PX.min(), PX.max(), PY.min(), PY.max(), PZ.min(), PZ.max()])
    mn, mx = lims.min(), lims.max()
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx); ax.set_zlim(mn, mx)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

def fig_sequence_projection(path="fig_sequence_projection.png", seed=7):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3,2))
    U_ref, _ = np.linalg.qr(A)

    g = np.linspace(-1.5, 1.5, 10)
    Gx, Gy = np.meshgrid(g, g)
    plane_pts = (U_ref @ np.vstack([Gx.ravel(), Gy.ravel()])).reshape(3, -1)
    PX = plane_pts[0].reshape(Gx.shape)
    PY = plane_pts[1].reshape(Gx.shape)
    PZ = plane_pts[2].reshape(Gx.shape)

    # synthetic 3D trajectory
    T = 60
    traj3d = np.cumsum(rng.normal(scale=0.15, size=(T, 3)), axis=0)
    # project to the plane coordinates and back to 3D on-plane curve
    Z2 = traj3d @ U_ref
    traj_proj3d = (U_ref @ Z2.T).T

    idx = np.linspace(0, T-1, 6, dtype=int)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(PX, PY, PZ, rstride=1, cstride=1, linewidth=0.6)
    ax.plot(traj3d[:,0], traj3d[:,1], traj3d[:,2], linewidth=1.6)                    # raw
    ax.plot(traj_proj3d[:,0], traj_proj3d[:,1], traj_proj3d[:,2], '--', linewidth=2) # projected
    for t in idx:
        x, xp = traj3d[t], traj_proj3d[t]
        ax.plot([x[0], xp[0]], [x[1], xp[1]], [x[2], xp[2]], linewidth=1.0)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Frames $\\mathbf{x}_t$ project to $U^\\top \\mathbf{x}_t$ (on-plane curve = k-D trajectory)')
    all_pts = np.vstack([np.column_stack([PX.ravel(), PY.ravel(), PZ.ravel()]), traj3d, traj_proj3d])
    mins = np.min(all_pts, axis=0); maxs = np.max(all_pts, axis=0)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

def fig_two_curves_k2(path="fig_two_kd_curves.png", seed=0):
    rng = np.random.default_rng(seed)
    T1, T2 = 60, 54
    c1 = np.cumsum(rng.normal(scale=0.12, size=(T1,2)), axis=0)
    c2 = np.cumsum(rng.normal(scale=0.12, size=(T2,2)), axis=0) + np.array([0.5, -0.2])

    plt.figure(figsize=(6,6))
    plt.plot(c1[:,0], c1[:,1], linewidth=2, label="curve A")
    plt.plot(c2[:,0], c2[:,1], '--', linewidth=2, label="curve B")
    # tiny arrows at starts
    plt.quiver(c1[0,0], c1[0,1], c1[1,0]-c1[0,0], c1[1,1]-c1[0,1], angles='xy', scale_units='xy', scale=1)
    plt.quiver(c2[0,0], c2[0,1], c2[1,0]-c2[0,0], c2[1,1]-c2[0,1], angles='xy', scale_units='xy', scale=1)
    plt.title("Two k-D curves (k=2) to be aligned by DTW")
    plt.xlabel("z1"); plt.ylabel("z2"); plt.axis('equal'); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

def fig_dtw_cost_path(path="fig_dtw_cost_path.png", seed=0):
    rng = np.random.default_rng(seed)
    T1, T2 = 60, 54
    c1 = np.cumsum(rng.normal(scale=0.12, size=(T1,2)), axis=0)
    c2 = np.cumsum(rng.normal(scale=0.12, size=(T2,2)), axis=0) + np.array([0.5, -0.2])

    def dtw_cost(Z, W):
        n, m = len(Z), len(W)
        C = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                C[i,j] = np.linalg.norm(Z[i]-W[j])
        D = np.full((n+1, m+1), np.inf); D[0,0]=0
        for i in range(1,n+1):
            for j in range(1,m+1):
                D[i,j] = C[i-1,j-1] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        i, j = n, m; path=[]
        while i>0 and j>0:
            path.append((i-1,j-1))
            step = np.argmin([D[i-1,j], D[i,j-1], D[i-1,j-1]])
            if step==0: i-=1
            elif step==1: j-=1
            else: i-=1; j-=1
        path.reverse()
        return C, np.array(path)

    C, path_idx = dtw_cost(c1, c2)
    plt.figure(figsize=(6,5))
    plt.imshow(C, origin='lower', aspect='auto')
    plt.plot(path_idx[:,1], path_idx[:,0], linewidth=2)
    plt.title("DTW cost matrix with optimal path")
    plt.xlabel("j (sequence 2)"); plt.ylabel("i (sequence 1)")
    plt.colorbar(label="per-step cost")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    fig_shared_subspace_plane()
    fig_sequence_projection()
    fig_two_curves_k2()
    fig_dtw_cost_path()
    print("Saved: fig_shared_subspace_plane.png")
    print("Saved: fig_sequence_projection.png")
    print("Saved: fig_two_kd_curves.png")
    print("Saved: fig_dtw_cost_path.png")
