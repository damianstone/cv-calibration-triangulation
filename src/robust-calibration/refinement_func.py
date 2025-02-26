def refine_and_deduplicate_lines(raw_lines, white_mask,
                                 refine_dist=6,
                                 angle_thresh_degs=0.75,
                                 dist_thresh=1.5):
    """
    Post-Hough refinement (Section 3.2.2):
      1) For each line, gather white pixels near it.
      2) Fit a new line via least squares.
      3) Remove duplicates if lines are nearly equal.

    Parameters:
      raw_lines: array of shape (N,4), each row = (x1, y1, x2, y2).
      white_mask: 8-bit binary image where 255 = potential line pixels.
      refine_dist: max pixel distance from line for including white pixels.
      angle_thresh_degs: maximum angle difference for duplicates.
      dist_thresh: maximum distance difference for duplicates.

    Returns:
      A list of final lines, each in (nx, ny, d) form:
        - (nx, ny) is a normalized line normal (magnitude = 1)
        - d is the line’s distance from origin, ensuring d >= 0
    """
    if raw_lines is None or len(raw_lines) == 0:
        return []

    refined_lines = []
    height, width = white_mask.shape[:2]

    # --- (1) & (2) Refine each line ---
    for (x1, y1, x2, y2) in raw_lines:
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1e-5:
            continue
        
        # Initial normal, distance
        nx_init, ny_init = dy / length, -dx / length
        d_init = nx_init*x1 + ny_init*y1

        # Collect white pixels near this line
        pts = []
        minx, maxx = max(0, min(x1, x2) - refine_dist), min(width-1, max(x1, x2) + refine_dist)
        miny, maxy = max(0, min(y1, y2) - refine_dist), min(height-1, max(y1, y2) + refine_dist)

        for yy in range(miny, maxy+1):
            for xx in range(minx, maxx+1):
                if white_mask[yy, xx] == 0:
                    continue
                dist_val = abs(nx_init*xx + ny_init*yy - d_init)
                if dist_val <= refine_dist:
                    pts.append((xx, yy))

        if len(pts) < 2:
            continue

        # Fit line y = m*x + c
        pts = np.array(pts, dtype=np.float32)
        xs, ys = pts[:, 0], pts[:, 1]
        A = np.vstack([xs, np.ones(len(xs))]).T
        m, c = np.linalg.lstsq(A, ys, rcond=None)[0]

        # Convert slope-intercept to normal form
        # eqn: y - m*x - c = 0 => m*x + (-1)*y + c=0
        # normal = (m, -1), length = sqrt(m^2 + 1)
        n_len = np.sqrt(m*m + 1)
        nx_ = m / n_len
        ny_ = -1.0 / n_len
        d_ = c / n_len

        # ensure d_ >= 0
        if d_ < 0:
            nx_, ny_, d_ = -nx_, -ny_, -d_

        refined_lines.append((nx_, ny_, d_))

    # --- (3) Remove duplicates ---
    final_lines = []
    angle_thresh = np.cos(np.deg2rad(angle_thresh_degs))

    for (nx_, ny_, d_) in refined_lines:
        is_duplicate = False
        for (fnx, fny, fd) in final_lines:
            dot = nx_*fnx + ny_*fny  # cos(angle between normals)
            if dot > angle_thresh:  # angle is small
                if abs(d_ - fd) < dist_thresh:
                    is_duplicate = True
                    break
        if not is_duplicate:
            final_lines.append((nx_, ny_, d_))

    return final_lines