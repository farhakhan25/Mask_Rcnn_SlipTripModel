import cv2
import numpy as np
import pandas as pd
from scipy import stats
import os
import sys
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from delaunay_triangulation.triangulate import scatter_vertices, delaunay

from myutils import extract_index_nparray, get_bounding_box, detect_landmarks, get_delauny_triangles, \
    get_delauny_triangles, \
    get_brutforce_match, get_delauny_triangles_md2, get_box_region, extract_contour_points, draw_box

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

font = cv2.FONT_HERSHEY_SIMPLEX

# Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow("frozen_inference_coco.pb", "mask_rnn_coco.pbtxt")

# Generate random colors
colors = np.random.randint(0, 255, (80, 3))

# Load image
fpath = ROOT_DIR + f"/Input_Video/video/trip1.mp4"
cap = cv2.VideoCapture(fpath)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)


height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width, height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fpath = ROOT_DIR + f"/results1/slip_prediction_1.avi"
out1 = cv2.VideoWriter(fpath, fourcc, 4.0, (width, height), isColor=True)

result_parameter = []
result_avgdis = []
result_absdiff = []
result_hog = []
class_label = []
f = 0
g = 0
while True:
    if f < 220:
        ret_ini, img_ini = cap.read()
        f += 1
        continue
    elif f >= 305:
        break
    elif f == 220:
        ret_ini, img_ini_ori = cap.read()
        img_ini = img_ini_ori.copy()
        height_ini, width_ini, _ = img_ini.shape

        # Create black image
        black_image_ini = np.zeros((height_ini, width_ini, 3), np.uint8)
        black_image_ini[:] = (100, 100, 0)

        # Detect objects from yolo
        blob_ini = cv2.dnn.blobFromImage(img_ini, swapRB=True)
        net.setInput(blob_ini)

        boxes_ini, masks_ini = net.forward(["detection_out_final", "detection_masks"])
        detection_count_ini = boxes_ini.shape[2]

        for i in range(detection_count_ini):
            box_ini = boxes_ini[0, 0, i]
            class_id_ini = box_ini[1]
            score_ini = box_ini[2]
            if (not int(class_id_ini) == 0) or score_ini < 0.9:
                continue

            ori_image_ini = draw_box(box_ini, img_ini_ori, width_ini, height_ini)

            img_ini, mask0_ini, roi_ini, roi_gray_ini, roi_b_ini, roi_mask_ini = get_bounding_box(i, box_ini, img_ini,
                                                                                                  black_image_ini,
                                                                                                  masks_ini, width_ini,
                                                                                                  height_ini,
                                                                                                  class_id_ini)
            mask_ini, roi_ini, landmarks_points_ini = detect_landmarks(mask0_ini, roi_b_ini)
            roi_ini, roi_mask_ini, image_1_ini, indexes_triangles_ini, triangle_ini, centriod_ini = get_delauny_triangles(
                landmarks_points_ini, roi_ini, roi_mask_ini)

            # Delauny Tringularization with method 2
            roi_ini_n = roi_ini.copy()
            roi_ini_n = get_box_region(box_ini, img_ini, width_ini, height_ini)

            # HOG
            fd_ini, hog_image_ini = hog(image_1_ini, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                        visualize=True)
            ho_image_rescaled_ini = exposure.rescale_intensity(hog_image_ini, in_range=(0, 0.02))
            result_hog.append(np.mean(fd_ini))

            # Brute Force
            contours_ini, _ = cv2.findContours(np.array(mask0_ini, np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            height_ini, width_ini, _ = roi_ini_n.shape
            spacing = 10
            scatter = 0.5
            vertices_ini = scatter_vertices(plane_width=width_ini, plane_height=height_ini, spacing=spacing,
                                            scatter=scatter)
            triangles_ini = delaunay(vertices=vertices_ini, delete_super_shared=False)
            cnt_len = [len(c) for c in contours_ini]
            max_len_cont = cnt_len.index(max(cnt_len))
            all_points_ini, key_points_all_ini, centroids_all_ini, centroids_all_kp_ini, roi_ini_n = extract_contour_points(
                triangles_ini, contours_ini[max_len_cont], landmarks=landmarks_points_ini, triangle_type="all",
                roi=roi_ini_n)

            # Initiate ORB detector
            orb = cv2.ORB_create()
            # kp_ini, des_ini = orb.compute(roi_ini_n, key_points_all_ini)
            kp_ini, des_ini = orb.compute(roi_ini_n, centroids_all_kp_ini)

            ## Flann Distance
            sift = cv2.SIFT_create()
            # kp = sift.detect(roi_ini_n, None)
            # kp_sift_ini, des_sift_ini = sift.compute(roi_ini_n, key_points_all_ini)
            kp_sift_ini, des_sift_ini = sift.compute(roi_ini_n, centroids_all_kp_ini)

            g += 1
    else:
        ret_fin, img_fin_ori = cap.read()
        img_fin_ori_gray = cv2.cvtColor(img_fin_ori, cv2.COLOR_BGR2GRAY)
        img_fin = img_fin_ori.copy()
        height_fin, width_fin, _ = img_fin.shape

        # Create black image
        black_image_fin = np.zeros((height_fin, width_fin, 3), np.uint8)
        black_image_fin[:] = (100, 100, 0)

        # Detect objects
        blob_fin = cv2.dnn.blobFromImage(img_fin, swapRB=True)
        net.setInput(blob_fin)

        boxes_fin, masks_fin = net.forward(["detection_out_final", "detection_masks"])
        detection_count_fin = boxes_fin.shape[2]
        for i in range(detection_count_fin):
            box_fin = boxes_fin[0, 0, i]
            class_id_fin = box_fin[1]
            score_fin = box_fin[2]
            # print("class info", class_id_fin, score_fin)
            if (not int(class_id_fin) == 0) or score_fin < 0.9:  # 0.95
                continue

            ori_image_fin = draw_box(box_fin, img_fin_ori, width_fin, height_fin)

            img_fin, mask0_fin, roi_fin, roi_gray_fin, roi_b_fin, roi_mask_fin = get_bounding_box(i, box_fin, img_fin,
                                                                                                  black_image_fin,
                                                                                                  masks_fin, width_fin,
                                                                                                  height_fin,
                                                                                                  class_id_fin)
            mask_fin, roi_fin, landmarks_points_fin = detect_landmarks(mask0_fin, roi_b_fin)
            roi_fin, roi_mask_fin, image_1_fin, indexes_triangles_fin, triangle_fin, centriod_fin = get_delauny_triangles(
                landmarks_points_fin, roi_fin, roi_mask_fin)

            roi_fin_n = get_box_region(box_fin, img_fin, width_fin, height_fin)

            # Brute Force
            contours_fin, _ = cv2.findContours(np.array(mask0_fin, np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            height_fin, width_fin, _ = roi_fin_n.shape
            vertices_fin = scatter_vertices(plane_width=width_fin, plane_height=height_fin, spacing=spacing,
                                            scatter=scatter)
            triangles_fin = delaunay(vertices=vertices_fin, delete_super_shared=False)

            cnt_len = [len(c) for c in contours_fin]
            max_len_cont = cnt_len.index(max(cnt_len))
            all_points_fin, key_points_all_fin, centroids_all_fin, centroids_all_kp_fin, roi_fin_n = extract_contour_points(
                triangles_fin, contours_fin[max_len_cont], landmarks=landmarks_points_fin, triangle_type="all",
                roi=roi_fin_n)

            kp_fin, des_fin = orb.compute(roi_fin_n, centroids_all_kp_fin)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_ini, des_fin)
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:int(len(matches) * .75)]
            print("len(matches) : ", len(matches))
            avg_dis = 0
            good_points = []
            for x in matches:
                avg_dis = avg_dis + x.distance
            avg_dis = avg_dis / len(matches)
            draw_params = dict(matchColor=(0, 0, 255), singlePointColor=(0, 255, 0), flags=2)
            matching_result_orb = cv2.drawMatches(roi_ini_n, kp_ini, roi_fin_n, kp_fin, matches, None, **draw_params)

            ## SIFT Distance
            # kp = sift.detect(roi_ini_n, None)
            # kp_sift_fin, des_sift_fin = sift.compute(roi_fin_n, key_points_all_fin)
            kp_sift_fin, des_sift_fin = sift.compute(roi_fin_n, centroids_all_kp_fin)

            FLAN_INDEX_KDTREE = 0
            index_params = dict(algorithm=0, trees=10)
            search_params = dict()
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            ## FLANN Matcher
            matches = flann.knnMatch(des_sift_ini, des_sift_fin, k=2)
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m1, m2) in enumerate(matches):
                if m1.distance < 0.75 * m2.distance:
                    matchesMask[i] = [1, 0]
            good_matches = []
            good_matches_d = []
            for m1, m2 in matches:
                if m1.distance < 0.95 * m2.distance:
                    good_matches.append([m1])
                    good_matches_d.append(m1.distance)
            draw_params = dict(matchColor=(0, 0, 255), singlePointColor=(0, 255, 0), matchesMask=matchesMask, flags=2)
            matching_result_flann = cv2.drawMatchesKnn(roi_ini_n, kp_sift_ini, roi_fin_n, kp_sift_fin, matches, None,
                                                       **draw_params)
            print(np.mean(good_matches_d), np.median(good_matches_d), stats.mode(good_matches_d))

            ## HOG
            fd_fin, hog_image_fin = hog(image_1_fin, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                        visualize=True)
            ho_image_rescaled_fin = exposure.rescale_intensity(hog_image_fin, in_range=(0, 0.02))
            ho_image_rescaled_fin_1 = ho_image_rescaled_fin.astype(np.uint8)
            result_hog.append(np.mean(fd_fin))

            # hog_amd = round(abs(np.mean(fd_fin)-np.mean(fd_ini))/np.mean(fd_ini), 2)
            hog_amd = abs(np.mean(fd_fin) - np.mean(fd_ini))
            result_avgdis.append(avg_dis)
            result_absdiff.append(hog_amd)

            title = "unknown"
            if not len(result_avgdis) == 0 and not len(result_hog) == 0:
                print(np.mean(result_avgdis) - np.std(result_avgdis), np.mean(result_avgdis),
                      np.mean(result_avgdis) + np.std(result_avgdis),
                      np.mean(result_avgdis) + 2 * np.std(result_avgdis),
                      np.mean(result_avgdis) + 3 * np.std(result_avgdis))
                print(np.mean(fd_fin), np.mean(result_hog) - np.std(result_hog), np.mean(result_hog),
                      np.mean(result_hog) + np.std(result_hog), np.mean(result_hog) + 2 * np.std(result_hog),
                      np.mean(result_hog) + 3 * np.std(result_hog))
                if (avg_dis > 48 and (
                        np.mean(fd_fin) > 0.06 or (hog_amd > 0.01))) or hog_amd > 0.01 or avg_dis > 50 or np.mean(
                        fd_fin) > 0.07:
                    title = "slip"
                    class_label.append("slip")
                    result_parameter.append([f, avg_dis, np.mean(fd_ini), np.mean(fd_fin), hog_amd, 'slip'])
                else:
                    title = "noslip"
                    class_label.append("no slip")
                    result_parameter.append([f, avg_dis, np.mean(fd_ini), np.mean(fd_fin), hog_amd, 'noslip'])

            df = pd.DataFrame(result_parameter, columns=['frame_id', "avg_brut_dist", "hog_mean_prev", "hog_mean_curnt",
                                                         "hog_abs_mean_diff", "class_label"])
            fpath = ROOT_DIR + f"/results1/slip_result_1.csv"
            df.to_csv(fpath, index=False)
            print(f"frame {f} :  {avg_dis},  {hog_amd}, {title}")

            cv2.putText(img_fin, title, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            # out1.write(img_fin)
            # ori_image_class = get_box_region(box_ini, img_fin, width_fin, height_fin)
            ori_image_class = ori_image_fin.copy()
            cv2.putText(ori_image_class, title, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            print(ori_image_class.shape)
            out1.write(ori_image_class)
            
        cv2.imshow("ori_image_fin", ori_image_fin)
        
        cv2.imshow("mask0_fin", mask0_fin)
        cv2.imshow("roi_fin", roi_fin)
        # cv2.imshow("roi_gray_fin", roi_gray_fin)
        # cv2.imshow("roi_mask_fin", roi_mask_fin)
        
        cv2.imshow("Image_ini", img_ini)
        cv2.imshow("Image_fin", img_fin)
        cv2.imshow("image_1_fin", image_1_fin)
        cv2.imshow("matching_result_orb", matching_result_orb)
        # cv2.imshow("matching_result_flann", matching_result_flann)
        cv2.imshow("ho_image_rescaled_fin", ho_image_rescaled_fin)
        #cv2.imshow("hog_image_fin", hog_image_fin)
        cv2.imshow("roi_fin_n", roi_fin_n)
        cv2.imshow("ori_image_class", ori_image_class)

        # cv2.imshow("image_1_ini", image_1_ini)
        # cv2.imshow("Image_ini", img_ini)
        # cv2.imshow("ho_image_rescaled_ini", ho_image_rescaled_ini)
        # cv2.imshow("roi_ini_n", roi_ini_n)

        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/ori/', f'frame{f}.jpg'), img_fin_ori)
        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/gray/', f'frame{f}.jpg'), img_fin_ori_gray)
        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/box/', f'frame{f}.jpg'), ori_image_fin)
        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/mask/', f'frame{f}.jpg'), image_1_fin)
        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/triangles/', f'frame{f}.jpg'), roi_fin_n)
        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/bf/', f'frame{f}.jpg'), matching_result_orb)
        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/hog/', f'frame{f}.jpg'), hog_image_fin)
        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/final/', f'frame{f}.jpg'), img_fin)
        cv2.imwrite(os.path.join(ROOT_DIR + '/results1/class/', f'frame{f}.jpg'), ori_image_class)


        img_ini = img_fin
        roi_ini_n = roi_fin_n
        mask0_ini = mask0_fin
        roi_ini = roi_fin
        roi_gray_ini = roi_gray_fin
        roi_b_ini = roi_b_fin
        roi_mask_ini = roi_mask_fin
        mask_ini = mask_fin
        roi_ini = roi_fin
        landmarks_points_ini = landmarks_points_fin
        roi_ini = roi_fin
        roi_mask_ini = roi_mask_fin
        image_1_ini = image_1_fin
        contours_ini = contours_fin
        vertices_ini = vertices_fin
        triangles_ini = triangles_fin
        all_points_ini = all_points_fin
        key_points_all_ini = key_points_all_fin
        kp_ini = kp_fin
        des_ini = des_fin
        fd_ini = fd_fin
        centroids_all_ini = centroids_all_fin
        centroids_all_kp_ini = centroids_all_kp_fin

        g += 1

    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    f += 1

cv2.waitKey(0)
cv2.destroyAllWindows()
out1.release()

print(result_parameter)
print(result_avgdis)
print(result_absdiff)

plt.plot(result_absdiff)
# plt.axhline(y = 0.02, color = 'black', linestyle = '-')
plt.draw()
plt.savefig("abs_hog_diff.png")
plt.show()
plt.pause(0.01)

plt.plot(result_avgdis)
# plt.axhline(y = 50, color = 'black', linestyle = '-')
plt.draw()
plt.savefig("avg_brute_dist.png")
plt.show()
plt.pause(0.01)
