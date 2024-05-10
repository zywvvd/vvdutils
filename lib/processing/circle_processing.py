import numpy as np
import cv2


def Geometric2Conic(ellipse):
    """
    Geometric to conic parameter conversion
    References
    ----
    Adapted from Swirski's ConicSection.h
    https://bitbucket.org/Leszek/pupil-tracker/
    """

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0, y0), (bb, aa), phi_b_deg = ellipse

    # Semimajor and semiminor axes
    a, b = aa / 2, bb / 2

    # Convert phi_b from deg to rad
    phi_b_rad = phi_b_deg * np.pi / 180.0

    # Major axis unit vector
    ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)

    # Useful intermediates
    a2 = a * a
    b2 = b * b

    #
    # Conic parameters
    #
    if a2 > 0 and b2 > 0:

        A = ax * ax / a2 + ay * ay / b2;
        B = 2 * ax * ay / a2 - 2 * ax * ay / b2;
        C = ay * ay / a2 + ax * ax / b2;
        D = (-2 * ax * ay * y0 - 2 * ax * ax * x0) / a2 + (2 * ax * ay * y0 - 2 * ay * ay * x0) / b2;
        E = (-2 * ax * ay * x0 - 2 * ay * ay * y0) / a2 + (2 * ax * ay * x0 - 2 * ax * ax * y0) / b2;
        F = (2 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2 + (
                -2 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0) / b2 - 1;

    else:

        # Tiny dummy circle - response to a2 or b2 == 0 overflow warnings
        A, B, C, D, E, F = (1, 0, 1, 0, 0, -1e-6)

    # Compose conic parameter array
    conic = np.array((A, B, C, D, E, F))

    return conic


def ConicFunctions(pnts, ellipse):
    """
    Calculate various conic quadratic curve support functions
    General 2D quadratic curve (biquadratic)
    Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    For point on ellipse, Q = 0, with appropriate coefficients
    Parameters
    ----
    pnts : n x 2 array of floats
    ellipse : tuple of tuples
    Returns
    ----
    distance : array of floats
    grad : array of floats
    absgrad : array of floats
    normgrad : array of floats
    References
    ----
    Adapted from Swirski's ConicSection.h
    https://bitbucket.org/Leszek/pupil-tracker/
    """

    # Suppress invalid values
    np.seterr(invalid='ignore')

    # Convert from geometric to conic ellipse parameters
    conic = Geometric2Conic(ellipse)

    # Row vector of conic parameters (Axx, Axy, Ayy, Ax, Ay, A1) (1 x 6)
    C = np.array(conic)

    # Extract vectors of x and y values
    x, y = pnts[:, 0], pnts[:, 1]

    # Construct polynomial array (6 x n)
    X = np.array((x * x, x * y, y * y, x, y, np.ones_like(x)))

    # Calculate Q/distance for all points (1 x n)
    distance = C.dot(X)

    # Quadratic curve gradient at (x,y)
    # Analytical grad of Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    # (dQ/dx, dQ/dy) = (2Ax + By + D, Bx + 2Cy + E)

    # Construct conic gradient coefficients vector (2 x 3)
    Cg = np.array(((2 * C[0], C[1], C[3]), (C[1], 2 * C[2], C[4])))

    # Construct polynomial array (3 x n)
    Xg = np.array((x, y, np.ones_like(x)))

    # Gradient array (2 x n)
    grad = Cg.dot(Xg)

    # Normalize gradient -> unit gradient vector
    # absgrad = np.apply_along_axis(np.linalg.norm, 0, grad)
    absgrad = np.sqrt(np.sqrt(grad[0, :] ** 2 + grad[1, :] ** 2))
    normgrad = grad / absgrad

    return distance, grad, absgrad, normgrad


# @nb.jit(nopython=True)
def outer_pts_collect(mag, ang):

    target_radius = np.percentile(mag, 95)
    m = target_radius
    round_step = 1.5 * np.pi / m

    inner_pts_indx = []
    outer_pts_indx = []
    mag_dict = {}
    indx_dict = {}

    for i, (a, m) in enumerate(zip(ang, mag)):
        av = a.item()
        mv = m.item()

        ar = int(round(av / round_step))
        t = mag_dict.get(ar, None)
        s = indx_dict.get(ar, None)

        if t is None:
            inner = mv
            outer = mv
            ind_inner = i
            ind_outer = i

        else:
            inner, outer = t
            ind_inner, ind_outer = s
            if mv < inner:
                inner = mv
                ind_inner = i
            elif mv > outer:
                outer = mv
                ind_outer = i

        mag_dict[ar] = (inner, outer)
        indx_dict[ar] = (ind_inner, ind_outer)

    for k, (i, o) in indx_dict.items():
        mi, mo = mag_dict[k]
        if mi == mo or (mo - mi) / target_radius < 0.3:
            continue
        else:
            inner_pts_indx.append(i)
            outer_pts_indx.append(o)

    return inner_pts_indx, outer_pts_indx


def EllipseError(pnts, ellipse):
    """
    Ellipse fit error function
    """

    # Suppress divide-by-zero warnings
    np.seterr(divide='ignore')

    # Calculate algebraic distances and gradients of all points from fitted ellipse
    distance, grad, absgrad, normgrad = ConicFunctions(pnts, ellipse)

    # Calculate error from distance and gradient
    # See Swirski et al 2012
    # TODO : May have to use distance / |grad|^0.45 - see Swirski source

    # Gradient array has x and y components in rows (see ConicFunctions)
    err = distance / absgrad

    return err


def EllipseNormError(pnts, ellipse):
    """
    Error normalization factor, alpha
    Normalizes cost to 1.0 at point 1 pixel out from minor vertex along minor axis
    """

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0, y0), (bb, aa), phi_b_deg = ellipse

    # Semiminor axis
    b = bb / 2

    # Convert phi_b from deg to rad
    phi_b_rad = phi_b_deg * np.pi / 180.0

    # Minor axis vector
    bx, by = np.cos(phi_b_rad), np.sin(phi_b_rad)

    # Point one pixel out from ellipse on minor axis
    p1 = np.array((x0 + (b + 1) * bx, y0 + (b + 1) * by)).reshape(1, 2)

    # Error at this point
    err_p1 = EllipseError(p1, ellipse)

    # Errors at provided points
    err_pnts = EllipseError(pnts, ellipse)

    return err_pnts / err_p1


def OverlayRANSACFit(img, all_pnts, inlier_pnts, ellipse):
    """
    NOTE
    ----
    All points are (x,y) pairs, but arrays are (row, col) so swap
    coordinate ordering for correct positioning in array
    """

    # Overlay all pnts in red
    for col, row in all_pnts:
        img[row, col] = [0, 0, 255]

    # Overlay inliers in green
    for col, row in inlier_pnts:
        img[row, col] = [0, 255, 0]

    # Overlay inlier fitted ellipse in yellow
    cv2.ellipse(img, ellipse, (0, 255, 255), 1)


def FitEllipse_RANSAC(pnts, roi=None, max_itts=5, max_refines=3, max_perc_inliers=95.0, graphics=False):
    '''
    Robust ellipse fitting to segmented boundary points
    Parameters
    ----
    pnts : n x 2 array of integers
        Candidate pupil-iris boundary points from edge detection
    roi : 2D scalar array
        Grayscale image of pupil-iris region for display only
    max_itts : integer
        Maximum RANSAC ellipse candidate iterations
    max_refines : integer
        Maximum RANSAC ellipse inlier refinements
    max_perc_inliers : float
        Maximum inlier percentage of total points for convergence
    Returns
    ----
    best_ellipse : tuple of tuples
        Best fitted ellipse parameters ((x0, y0), (a,b), theta)
    '''

    rnd = np.random.RandomState(123)

    # Debug flag
    DEBUG = False

    # Output flags
    # graphics = cfg.getboolean('OUTPUT', 'graphics')

    # Suppress invalid values
    np.seterr(invalid='ignore')

    # Maximum normalized error squared for inliers
    max_norm_err_sq = 4.0

    # Tiny circle init
    best_ellipse = ((0, 0), (1e-6, 1e-6), 0)

    # Count pnts (n x 2)
    n_pnts = pnts.shape[0]

    LS_ellipse = cv2.fitEllipse(pnts)

    # return LS_ellipse, []

    norm_err = EllipseNormError(pnts, LS_ellipse)
    ok_point_index = np.abs(norm_err) < np.percentile(np.abs(norm_err), 90)
    pnts = pnts[ok_point_index, :]

    # Break if too few points to fit ellipse (RARE)
    if n_pnts < 5:
        return best_ellipse, None

    best_perc_inliers = 0
    best_inlier_pnts = pnts
    # Ransac iterations

    sample_num = 90
    for itt in range(0, max_itts):
        # Select points
        if len(pnts) < sample_num:
            sample_pnts = np.asarray(list(pnts))
        else:
            inds = rnd.choice(range(len(pnts)), sample_num, replace=False)
            sample_pnts = pnts[inds,:]

        # Fit ellipse to points
        ellipse = cv2.fitEllipse(sample_pnts)

        # Refine inliers iteratively
        for refine in range(0, max_refines):

            # Calculate normalized errors for all points
            norm_err = EllipseNormError(pnts, ellipse)

            # Identify inliers
            inliers = np.nonzero(norm_err ** 2 < max_norm_err_sq)[0]

            # Update inliers set
            inlier_pnts = pnts[inliers]

            # Protect ellipse fitting from too few points
            if inliers.size < 5:
                if DEBUG: print('Break < 5 Inliers (During Refine)')
                break

            # Fit ellipse to refined inlier set
            ellipse = cv2.fitEllipse(inlier_pnts)

        # End refinement

        # Count inliers (n x 2)
        n_inliers = inliers.size
        perc_inliers = (n_inliers * 100.0) / n_pnts

        # Update best ellipse
        if best_perc_inliers < perc_inliers:
            best_perc_inliers = perc_inliers
            best_ellipse = ellipse
            best_inlier_pnts = inlier_pnts

        if perc_inliers > max_perc_inliers:
            if DEBUG: print('Break Max Perc Inliers')
            break

    return best_ellipse, best_inlier_pnts

# https://zhuanlan.zhihu.com/p/82184417
def get_ellipse_param(major_radius, minor_radius, radian):
    '''
    根据椭圆的主轴和次轴半径以及旋转角度(默认圆心在原点), 得到椭圆参数方程的参数, 
    椭圆参数方程为:
        A * x^2 + B * x * y + C * y^2 + F = 0
    '''
    a, b = major_radius, minor_radius
    sin_theta = np.sin(-radian)
    cos_theta = np.cos(-radian)
    A = a**2 * sin_theta**2 + b**2 * cos_theta**2
    B = 2 * (a**2 - b**2) * sin_theta * cos_theta
    C = a**2 * cos_theta**2 + b**2 * sin_theta**2
    F = -a**2 * b**2
    return A, B, C, F


def calculate_rectangle(A, B, C, F):
    '''
    根据椭圆参数方程的参数, 得到椭圆的外接矩形top-left和right-bottom坐标。
    '''

    # 椭圆上下外接点的纵坐标值
    y = np.sqrt(4*A*F / (B**2 - 4*A*C))
    y1, y2 = -np.abs(y), np.abs(y)
    
    # 椭圆左右外接点的横坐标值
    x = np.sqrt(4*C*F / (B**2 - 4*C*A))
    x1, x2 = -np.abs(x), np.abs(x)
    
    return (x1, y1), (x2, y2)


def get_rectangle(center_x, center_y, major_radius, minor_radius, radian):
    '''
    按照数据集接口返回矩形框
    center_x:中心点横坐标
    center_y:中心点纵坐标
    major_radius:主轴的半径
    minor_radius:短轴半径
    angle:(顺时针)旋转角度
    '''
    A, B, C, F = get_ellipse_param(major_radius, minor_radius, radian)
    p1, p2 = calculate_rectangle(A, B, C, F)
    return (center_x+p1[0], center_y+p1[1]), (center_x+p2[0], center_y+p2[1])

def draw_ellipse(img, center_xy, major_axis, minor_axis, angle=0, color=[255, 255, 255], thickness=1):
    x, y = center_xy
    res_img = cv2.ellipse(img.copy(), [[x, y], [major_axis, minor_axis], angle],  color=color, thickness=thickness)
    return res_img

def ellipse2bbox(ellipse):
    (center_x, center_y), (major, minor), angle = ellipse
    major_radius = major / 2
    minor_radius = minor / 2
    radian = angle * np.pi / 180 
    rectangle = get_rectangle(center_x, center_y, major_radius, minor_radius, radian)
    ((a, b), (c, d)) = rectangle
    bbox = [min(a, c), min(b, d), max(a, c), max(b, d)]
    return bbox

def get_cycle_num(signal):
    """获取环境数据信号周期数

    """
    def gain(data, move_matrix):
        value_matrix = move_matrix
        weight_matrix = np.exp(- np.square(move_matrix - data))
        processed_signal = np.sum(value_matrix * weight_matrix, axis=1)
        return processed_signal

    def construct_circle_matrix(data):
        move_matrix = np.zeros([len(data)] * 2)
        for index in range(len(data)):
            new_data = np.hstack([data[index:], data[:index]])
            move_matrix[index, ...] = new_data
        return move_matrix

    def self_con(data):
        data = data - data.min()
        move_matrix = construct_circle_matrix(data)
        processed_signal = gain(data, move_matrix)

        fft_res = fft(processed_signal - processed_signal.mean())
        mt.PIS(abs(fft_res))
        cycle = np.argmax(abs(fft_res))

        return cycle
    
    return self_con(signal)