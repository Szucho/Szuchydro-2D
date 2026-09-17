import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


SAVE_OUTPUTS = True
FPS = 30



def read_meta(outdir):
    meta = {}
    with open(os.path.join(outdir, "meta.txt")) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 2:
                meta[parts[0]] = float(parts[1])
    return meta


def read_grid(path):
    """Returns (blocks, frames).
    blocks: list of dicts, one per MeshBlock, in order
      {level, lx1, lx2, x1min, x1max, x2min, x2max, nx1, nx2}
    frames: list of (t, block_arrays), block_arrays[k] has shape
      (blocks[k]['nx1'], blocks[k]['nx2'], 4)
    that block's own active state vector, same (i,j) order the sim used."""
    with open(path, 'rb') as f:
        nblocks = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        blocks = []
        for _ in range(nblocks):
            level = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            lx1, lx2 = np.frombuffer(f.read(16), dtype=np.int64)
            x1min, x1max, x2min, x2max = np.frombuffer(f.read(32), dtype=np.float64)
            nx1, nx2 = np.frombuffer(f.read(16), dtype=np.uint64)
            blocks.append(dict(level=level, lx1=int(lx1), lx2=int(lx2),
                                x1min=x1min, x1max=x1max, x2min=x2min, x2max=x2max,
                                nx1=int(nx1), nx2=int(nx2)))

        frames = []
        while True:
            tbuf = f.read(8)
            if not tbuf: break
            t = np.frombuffer(tbuf, dtype=np.float64)[0]
            block_arrays = []
            complete = True
            for blk in blocks:
                n = blk['nx1'] * blk['nx2'] * 4
                raw = f.read(n * 8)
                if len(raw) != n * 8:
                    complete = False
                    break
                block_arrays.append(np.frombuffer(raw, dtype=np.float64).reshape(blk['nx1'], blk['nx2'], 4))
            if not complete: break
            frames.append((t, block_arrays))
    return blocks, frames


def build_canvas_layout(blocks):
    """
    THIS SOLUTION MIGHT NOT BE THE BEST!!!
    Work out a single finest-resolution grid every block's data can be
    pasted into. Returns (nx_c, ny_c, placements) where placements[k] is
    (i0, i1, j0, j1, upsample_i, upsample_j) for blocks[k] the pixel
    range it occupies on the canvas, and how much to repeat its own
    cells by along each axis to fill that range."""
    maxlevel = max(b['level'] for b in blocks)
    #my reasoning here:
    #every block has the same active cell count per axis by construction
    #(MeshInputs::nx1_block/nx2_block), so the canvas size is just that
    #times the number of level-0 blocks needed to tile the domain, times
    #the finest refinement factor
    nx1_block = blocks[0]['nx1']
    nx2_block = blocks[0]['nx2']
    #infer root block-grid dimensions from the level-0 blocks' lx1/lx2 span
    lvl0 = [b for b in blocks if b['level'] == 0]
    nblk1_root = max(b['lx1'] for b in lvl0) + 1
    nblk2_root = max(b['lx2'] for b in lvl0) + 1

    nx_c = nblk1_root * nx1_block * (1 << maxlevel)
    ny_c = nblk2_root * nx2_block * (1 << maxlevel)

    placements = []
    for b in blocks:
        factor = 1 << (maxlevel - b['level'])
        i0 = b['lx1'] * nx1_block * factor
        j0 = b['lx2'] * nx2_block * factor
        i1 = i0 + nx1_block * factor
        j1 = j0 + nx2_block * factor
        placements.append((i0, i1, j0, j1, factor, factor))
    return nx_c, ny_c, placements


def stitch_frame(block_arrays, placements, nx_c, ny_c):
    canvas = np.empty((nx_c, ny_c, 4))
    for data, (i0, i1, j0, j1, fi, fj) in zip(block_arrays, placements):
        canvas[i0:i1, j0:j1, :] = np.repeat(np.repeat(data, fi, axis=0), fj, axis=1)
    return canvas


def compute_primitives(data, gamma=1.4):
    rho = data[:, :, 0]
    u   = data[:, :, 1] / rho
    v   = data[:, :, 2] / rho
    E   = data[:, :, 3]
    p   = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    return rho, u, v, p


outdir = input("Output directory: ").strip()
meta = read_meta(outdir)

xmin, xmax = meta['xmin'], meta['xmax']
ymin, ymax = meta['ymin'], meta['ymax']
gamma      = meta['gamma']

blocks, frames = read_grid(os.path.join(outdir, "grid.bin"))
if not frames:
    raise RuntimeError("grid.bin is empty or could not be read")

maxlevel = max(b['level'] for b in blocks)
if len(blocks) > 1 or maxlevel > 0:
    print(f"grid.bin: {len(blocks)} block(s), refinement up to level {maxlevel}\\"
          f"stitching onto a single canvas for plotting (coarser blocks are "
          f"nearest-neighbor upsampled, not interpolated; this is a display "
          f"convenience, not a claim about solution accuracy there. PROPER PLOT IDEA NEEDED.).")

nx_c, ny_c, placements = build_canvas_layout(blocks)

times = [f[0] for f in frames]
prims = [compute_primitives(stitch_frame(f[1], placements, nx_c, ny_c), gamma) for f in frames]

#physical cell-center coordinates on the canvas
dx = (xmax - xmin) / nx_c
dy = (ymax - ymin) / ny_c
x_centers = xmin + (np.arange(nx_c) + 0.5) * dx
j_mid = ny_c // 2
y_mid_val = ymin + (j_mid + 0.5) * dy

#meshgrid boundaries for 2D pcolormesh
x_edges = np.linspace(xmin, xmax, nx_c + 1)
y_edges = np.linspace(ymin, ymax, ny_c + 1)
X, Y = np.meshgrid(x_edges, y_edges)

#1D slice animation
fig_1d, ax1 = plt.subplots(figsize=(9, 4.5))

line_rho, = ax1.plot([], [], 'b-',  lw=1.8, label=r'$\rho(x, y=0.5)$')
line_u,   = ax1.plot([], [], 'r--', lw=1.8, label=r'$u(x, y=0.5)$')
line_p,   = ax1.plot([], [], 'g-.', lw=1.8, label=r'$p(x, y=0.5)$')

ax1.set_xlim(xmin, xmax)
ax1.set_ylim(-0.05, 1.15)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('Primitive vals', fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc='center right', fontsize=11)

def update_1d(n):
    rho, u, _, p = prims[n]
    line_rho.set_data(x_centers, rho[:, j_mid])
    line_u.set_data(x_centers, u[:, j_mid])
    line_p.set_data(x_centers, p[:, j_mid])
    ax1.set_title(f"shock tube snippet primitive vals at t~{times[n]:.2f}", fontsize=12)
    return line_rho, line_u, line_p

ani_1d = animation.FuncAnimation(fig_1d, update_1d, frames=len(prims), interval=40, blit=False)

if SAVE_OUTPUTS:
    gif_1d_path = os.path.join(outdir, "shock_tube_1d_animation.gif")
    ani_1d.save(gif_1d_path, writer='pillow', fps=FPS)
    print(f"Saved animation: {gif_1d_path}")

#snippet of shock tube at tf
rho_f, u_f, _, p_f = prims[-1]

fig_snap, ax_snap = plt.subplots(figsize=(9, 4.5))
ax_snap.plot(x_centers, rho_f[:, j_mid], 'b-',  lw=1.8, label=r'$\rho(x, y=0.5)$')
ax_snap.plot(x_centers, u_f[:, j_mid],   'r--', lw=1.8, label=r'$u(x, y=0.5)$')
ax_snap.plot(x_centers, p_f[:, j_mid],   'g-.', lw=1.8, label=r'$p(x, y=0.5)$')

ax_snap.set_xlim(xmin, xmax)
ax_snap.set_ylim(-0.05, 1.15)
ax_snap.set_xlabel('x', fontsize=11)
ax_snap.set_ylabel(f'Primitive vals HLLC t={times[-1]:.4f}', fontsize=11)
ax_snap.grid(True, linestyle=":", alpha=0.6)
ax_snap.legend(loc='center right', fontsize=11)
ax_snap.set_title(f"shock tube snippet primitive vals at t~{times[-1]:.1f}", fontsize=12)

fig_snap.tight_layout()
fig_snap.savefig(os.path.join(outdir, "shock_tube_1d_snippet.png"), dpi=300)

if SAVE_OUTPUTS:
    snap_path = os.path.join(outdir, "shock_tube_1d_snippet.png")
    fig_snap.savefig(snap_path, dpi=300)
    print(f"Saved figure: {snap_path}")

#2D multi-panel animation of primitive vals
fig_2d, axes = plt.subplots(2, 2, figsize=(10, 5))
fig_2d.subplots_adjust(hspace=0.4, wspace=0.3)

titles = [r"$\rho$", r"$u$", r"$v$", r"$p$"]
cmaps  = ["viridis", "coolwarm", "coolwarm", "plasma"]

limits = [
    (min(p[0].min() for p in prims), max(p[0].max() for p in prims)),
    (min(p[1].min() for p in prims), max(p[1].max() for p in prims)),
    (min(p[2].min() for p in prims), max(p[2].max() for p in prims)),
    (min(p[3].min() for p in prims), max(p[3].max() for p in prims))
]

meshes = []
for ax, field, title, cmap, lim in zip(axes.flat, prims[0], titles, cmaps, limits):
    mesh = ax.pcolormesh(X, Y, field.T, cmap=cmap, vmin=lim[0], vmax=lim[1], shading='flat')
    ax.set_aspect('auto')
    fig_2d.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    meshes.append(mesh)

suptitle = fig_2d.suptitle(f"t = {times[0]:.4f}", fontsize=12, fontweight='bold')

def update_2d(n):
    rho, u, v, p = prims[n]
    for mesh, field in zip(meshes, [rho, u, v, p]):
        mesh.set_array(field.T.ravel())
    suptitle.set_text(f"t = {times[n]:.4f}")
    return meshes + [suptitle]

ani_2d = animation.FuncAnimation(fig_2d, update_2d, frames=len(prims), interval=40, blit=False)

if SAVE_OUTPUTS:
    gif_2d_path = os.path.join(outdir, "shock_tube_2d_animation.gif")
    ani_2d.save(gif_2d_path, writer='pillow', fps=FPS)
    print(f"Saved animation: {gif_2d_path}")

plt.show()
