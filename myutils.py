import cv2
import numpy as np
from delaunay_triangulation.triangulate import scatter_vertices, delaunay

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def draw_box(box, img, width, height):
    # Get box Coordinates
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
    return img

def get_box_region(box, img, width, height):
    # Get box Coordinates
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    roi = img[y: y2, x: x2]
    return roi

def get_bounding_box(i, box, img, black_image, masks, width, height, class_id):
    # Get box Coordinates
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    roi = img[y: y2, x: x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_mask = np.zeros_like(roi_gray)
    roi_b = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi_b.shape

    # Get the mask
    mask = masks[i, int(class_id)]
    mask = cv2.resize(mask, (roi_width, roi_height))
    _, mask = cv2.threshold(mask, .2, 255, cv2.THRESH_BINARY)

    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)

    return img, mask, roi, roi_gray, roi_b, roi_mask

def detect_landmarks(mask, roi):

    # Get mask coordinates
    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.fillPoly(roi, [cnt], (255, 0, 0))
        landmarks_points = []
        for pnt in cnt:
            x = int(pnt[0][0])
            y = int(pnt[0][1])
            landmarks_points.append((x, y))
            cv2.circle(roi, (x, y), 1, (0, 255, 0), -1)

        return mask, roi, landmarks_points

def get_delauny_triangles(landmarks_points, roi, roi_mask):
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.polylines(roi, [convexhull], True, (0, 0, 255), 3)
    cv2.fillConvexPoly(roi_mask, convexhull, 255)

    image_1 = cv2.bitwise_and(roi, roi, mask=roi_mask)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    triangle = []
    centriod = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            tri = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(tri)
            triangle.append([pt1, pt2, pt3])

            centd = ((pt1[0] + pt2[0] + pt3[0]) // 3, (pt1[1] + pt2[1] + pt3[1]) // 3)
            centriod.append(centd)

            # Drawing the centroid on the window
            cv2.circle(roi, centd, 4, (0, 255, 0))

            cv2.line(roi, pt1, pt2, (255, 0, 255), 2)
            cv2.line(roi, pt2, pt3, (255, 0, 255), 2)
            cv2.line(roi, pt1, pt3, (255, 0, 255), 2)

    return roi, roi_mask,  image_1, indexes_triangles, triangle, centriod

#voronoi diagrams or thiessen polygons
#save image ==> image cropping of image from bounding box to green outer line
def get_delauny_triangles_md2(image):
    height, width, _ = image.shape
    spacing = 10
    scatter = 0.5

    vertices = scatter_vertices(
        plane_width=width,
        plane_height=height,
        spacing=spacing,
        scatter=scatter
    )

    triangles = delaunay(
        vertices=vertices,
        delete_super_shared=False,
    )

    points = []
    for t in triangles:
        dummy = []
        x1 = int(t.vertices[0].x)
        y1 = int(t.vertices[0].y)
        pt1 = (x1, y1)
        dummy.append(pt1)

        x2 = int(t.vertices[1].x)
        y2 = int(t.vertices[1].y)
        pt2 = (x2, y2)
        dummy.append(pt2)

        x3 = int(t.vertices[2].x)
        y3 = int(t.vertices[2].y)
        pt3 = (x3, y3)
        dummy.append(pt3)
        points.append(dummy)

    for pnt in points:
        cv2.line(image, pnt[0], pnt[1], (255, 0, 255), 1)
        cv2.line(image, pnt[1], pnt[2], (255, 0, 255), 1)
        cv2.line(image, pnt[0], pnt[2], (255, 0, 255), 1)

        return image, points, vertices, triangles


# sift-surf
# image hashing
# FLANN in place of brute force, keep HOG as is, let us compare output of brute force matching with FLANN matching
# image differencing
# KNN or FLANN (region based study) - trigulated image - 128-164== 200
# Brute -> KNN FLANN (SHIFT-SURF) 0r image hashing
def get_brutforce_match(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = []
    avg_dis = 0
    for x in matches:
        avg_dis = avg_dis + x.distance
    avg_dis = avg_dis / len(matches)

    return kp1, des1, kp2, des2, matches, avg_dis


def extract_contour_points(triangles, contour, landmarks=None, triangle_type="all", roi=None):
    all_points = []
    boundary_points = []
    inside_points = []
    key_points_all = []
    key_points_boundary = []
    key_points_inside = []
    centroids_boundary = []
    centroids_inside = []
    centroids_all = []
    centroids_boundary_kp = []
    centroids_inside_kp = []
    centroids_all_kp = []
    for i, t in enumerate(triangles):
        intersect = [False, False, False]
        dummy = []
        dummy_keypoints = []
        cent_x=0.
        cent_y=0.
        for j, v in enumerate(t.vertices):
            pt = (v.x, v.y)
            cent_x = cent_x+v.x
            cent_y = cent_y+v.y
            result = cv2.pointPolygonTest(contour, pt, False)
            if result == 0:
                intersect[j] = True
                dummy.append((int(v.x), int(v.y)))
                kpt = cv2.KeyPoint(x=int(v.x), y=int(v.y), _size=1)
                dummy_keypoints.append(kpt)
            elif result == 1:
                intersect[j] = True
                dummy.append((int(v.x), int(v.y)))
                kpt = cv2.KeyPoint(x=int(v.x), y=int(v.y), _size=1)
                dummy_keypoints.append(kpt)
            else:
                intersect[j] = False
                dummy.append((int(v.x), int(v.y)))
                kpt = cv2.KeyPoint(x=int(v.x), y=int(v.y), _size=1)
                dummy_keypoints.append(kpt)

        cent_x=cent_x//3
        cent_y=cent_y//3
        if triangle_type=="boundary":
            if sum(intersect) > 0 and sum(intersect) < 3:
                print("Points on the boundries the contour!")
                boundary_points.append(dummy)
                key_points_boundary = key_points_boundary + dummy_keypoints
                centroids_boundary.append((cent_x, cent_y))
                kpt = cv2.KeyPoint(x=int(cent_x), y=int(cent_y), _size=1)
                centroids_boundary_kp.append(kpt)

                if not roi is None:
                    cv2.line(roi, dummy[0], dummy[1], (255, 0, 0), 1)
                    cv2.line(roi, dummy[1], dummy[2], (255, 0, 0), 1)
                    cv2.line(roi, dummy[0], dummy[2], (255, 0, 0), 1)
        elif triangle_type == "inside":
            if all(intersect):
                inside_points.append(dummy)
                key_points_inside = key_points_inside + dummy_keypoints
                centroids_inside.append((cent_x, cent_y))
                kpt = cv2.KeyPoint(x=int(cent_x), y=int(cent_y), _size=1)
                centroids_inside_kp.append(kpt)

                if not roi is None:
                    cv2.line(roi, dummy[0], dummy[1], (255, 0, 0), 1)
                    cv2.line(roi, dummy[1], dummy[2], (255, 0, 0), 1)
                    cv2.line(roi, dummy[0], dummy[2], (255, 0, 0), 1)
        else:
            if np.any(intersect):
                all_points.append(dummy)
                key_points_all = key_points_all + dummy_keypoints
                centroids_inside.append((cent_x, cent_y))
                kpt = cv2.KeyPoint(x=int(cent_x), y=int(cent_y), _size=1)
                centroids_all_kp.append(kpt)

                if not roi is None:
                    cv2.line(roi, dummy[0], dummy[1], (0, 255, 0), 1)
                    cv2.line(roi, dummy[1], dummy[2], (0, 255, 0), 1)
                    cv2.line(roi, dummy[0], dummy[2], (0, 255, 0), 1)

    if triangle_type == "boundary":
        if not landmarks is None:
            for x, y in landmarks:
                kpt = cv2.KeyPoint(x=int(x), y=int(y), _size=1)
                key_points_boundary.append(kpt)
                centroids_boundary_kp.append(kpt)
        return boundary_points, key_points_boundary, centroids_boundary, centroids_boundary_kp, roi
    elif triangle_type == "inside":
        if not landmarks is None:
            for x, y in landmarks:
                kpt = cv2.KeyPoint(x=int(x), y=int(y), _size=1)
                key_points_inside.append(kpt)
                centroids_inside_kp.append(kpt)
        return inside_points, key_points_inside, centroids_inside, centroids_inside_kp, roi
    else:
        if not landmarks is None:
            for x, y in landmarks:
                kpt = cv2.KeyPoint(x=int(x), y=int(y), _size=1)
                key_points_all.append(kpt)
                centroids_all_kp.append(kpt)
        return all_points, key_points_all, centroids_all, centroids_all_kp, roi

