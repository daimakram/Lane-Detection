#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math

import cv2
import numpy as np
import loguru
from pandas.compat import pa_version_under1p0
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

LOG = loguru.logger


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        self._cfg = cfg

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=self._cfg.POSTPROCESS.DBSCAN_EPS, min_samples=self._cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            LOG.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []
        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        return mask, lane_coords


import copy

def avg(ptr): # check this... why 3... wont it be a_len  
  a = copy.deepcopy(ptr)
  a_len = len(a)
  try:
    for i in range(1, a_len):
      a[0][0] += a[i][0]
      a[0][1] += a[i][1]
      a[0][2] += a[i][2]

    a[0][0] /= 3
    a[0][1] /= 3
    a[0][2] /= 3

    return a[0]
  except:
    return a

def find_vanishing(pts):
  print("\n")
  temp = []
  points = []
  arr = copy.deepcopy(pts)
  for i in arr: # get 2 points for each lane 
    i[0].append(1)
    i[10].append(1)
    temp.append( [i[0], i[10]] )

  # print(temp)

  for i in range(len(temp)): # get line for each lane 
    temp[i] = np.cross(temp[i][0],temp[i][1])

  # print(temp)

  for i in range(len(temp)-1): # get line for each lane 
    points.append( np.cross(temp[i], temp[i+1]) )
  
  vanishing_point = avg(points)
  try: # where is there a need for try except here?
    vanishing_point[0] = vanishing_point[0]/vanishing_point[2]
    vanishing_point[1] = vanishing_point[1]/vanishing_point[2]
    vanishing_point[2] = vanishing_point[2]/vanishing_point[2]

    print("point is: ", vanishing_point)

    return vanishing_point
  except:
    return vanishing_point

def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return k//ncol, k%ncol

def find_closest(y1, y2, a, b):
  y1 = np.array(y1)
  y2 = np.array(y2)
  y1 = y1.astype(int)
  y2 = y2.astype(int)
  a = np.array(a)
  b = np.array(b)
  a = a.astype(int)
  b = b.astype(int)

  print("inside")
  print(a[-5:,-5:])
  print(b[-5:,-5:])

  temp = np.zeros((y1.shape[0], y2.shape[0]))
  indices = []

  for i in range(y1.shape[0]): # difference of each point of y1 with y2
    temp[i] = abs(y1[i] - y2)

  # shp = temp.shape[0] // 4 # only look at starting points 
  # temp = temp[:shp, :]

  # print("new")
  # print(find_min_idx(temp))
  # print(np.min(temp))
  # print( np.partition(temp, 3)[:2, 0])
  # print( np.argpartition(temp, 3)[:2, 0])


  # print("temp: ", temp)
  smallest = np.zeros((temp.shape[0],)) # stores min value per y1 value
  yi = np.zeros((temp.shape[0],)) # stores index of min value 

  # assuming we will have atleast 10 points per lane 

  shp = temp.shape[0]
  # if shp > 30:
  #   shp = shp//4
  last_25pts = int(0.70*shp)
  last_15pts = int(0.90*shp)

  for i in range(temp.shape[0]):
    smallest[i] = np.min(temp[i])
    yi[i] = temp[i].argmin()

  print("len",len(smallest))
  #p1 = smallest.argmin()
  #p2 = smallest[shp//2:shp//2 + shp//4,].argmin() + shp//2

  p1 = smallest[last_25pts: last_15pts,].argmin() + last_25pts
  p2 = smallest[last_15pts:,].argmin() + last_15pts
  print("p1 p2")
  print(p1)
  print(p2)
  # print(smallest[:10,])
  # print(smallest[5:10,])
  # print(shp)
  # print(p1, smallest[p1,])
  # print(p2, smallest[p2,])

  # print(temp[:shp, :shp])
  # print(smallest[:shp,])
  # print(np.argpartition(smallest[:shp, ], shp-1))
  # least_diff = np.argpartition(smallest[:shp, ], shp-1)

  # print(smallest[:shp, ].argmin())
  # smallest[smallest[:shp, ].argmin()] = 9999
  # print(smallest[:shp, ].argmin())

  # print(least_diff)
  # print(yi[least_diff[0]])
  v1 = a[p1]
  v2 = b[ int(yi[p1]) ]
  indices.append([v1, v2])

  # print("v1 ",  v1)
  # l_index = -1
  # for i in range(1, len(least_diff)): # so y is different enough between the 2 points 
  #   if abs(least_diff[0] - least_diff[i]) > 10: # make this more general
  #     l_index = least_diff[i]
  #     break

  v1 = a[p2]
  v2 = b[ int(yi[p2]) ]
  indices.append([v1, v2])
  # print("v1 ",  v1)
  # min1 = smallest.argmin()
  # ymin = yi[min1]
  # print(min1, y1[min1], y2[ymin], a[min1], b[ymin])
  # indices.append([a[min1], b[ymin]])
  
  # smallest = np.delete(smallest, min1)
  # min2 = smallest.argmin()
  # ymin = yi[min2]
  # print(min2, y1[min2], y2[min2], a[min2], b[min2])
  # indices.append([a[min2+1], b[min2+1]])
  # print(smallest[:5,])
  # print(smallest.argmin(), smallest[smallest.argmin()])
  # print(y1[smallest.argmin()], y2[smallest.argmin()])

  # for i in range(temp.shape[0]):
  #   t2 = temp[i].argmin()
  #   indices.append([a1[i], a2[t2]])

  # indices = np.array(indices)
  # smt = np.argpartition(smallest, 5)
  # print("smallest: ", np.argpartition(smallest, 5, axis=-1))
  # print("smallest: ", smallest[np.argpartition(smallest, 5, axis=-1)])

  # sa = smt[:5,]
  # print("smt ", smt[:5],)
  # print("smallest ", smallest[smt[:5]],)
  # print("sa ", sa)
  # sm = min(sa)
  # sx = max(sa)

  # indices.append([y1[smt[0]], y2[smt[0]]])
  # indices.append([y1[smt[1]], y2[smt[1]]])

  
  if(indices[0][0][1] > indices[1][0][1]): # sorting acc to y 
    temp1 = indices[0]
    indices[0] = indices[1]
    indices[1] = temp1

  if(indices[0][0][0] > indices[0][1][0]): # sorting acc to y 
    temp1 = indices[0][0][0]
    indices[0][0][0] = indices[0][1][0]
    indices[0][1][0] = temp1
  if(indices[1][0][0] > indices[1][1][0]): # sorting acc to y 
    temp1 = indices[1][0][0]
    indices[1][0][0] = indices[1][1][0]
    indices[1][1][0] = temp1
  
  print("returning ", indices)

  return indices

def make_rect(ptr):
  arr = copy.deepcopy(ptr)
  final_pts = []
  
  #y1=(arr[0][0][1]+arr[0][1][1])//2
  #y2=(arr[1][0][1]+arr[1][0][1])//2
  #x1=(arr[0][0][0]+arr[0][1][0])//2
  #x2=(arr[1][0][0]+arr[1][0][0])//2
  #x1_diff=abs(arr[0][0][0]-arr[0][1][0])
  #x2_diff=abs(arr[1][0][0]+arr[1][0][0])

  #p1=[x1-x1_diff,y1]
  #p3=[x2-x2_diff,y2]
  #p2=[x1+x1_diff,y1]
  #p4=[x2+x2_diff,y2]
  
  print(arr)

  # x1_avg = (arr[0][0][0] + arr[0][1][0])//2  # p1x + p2x 
  # x2_avg = (arr[1][0][0] + arr[1][1][0])//2  # p1x + p2x 
  y_avg = (arr[0][0][1] + arr[0][1][1])//2 # p1y + p2y 
  dy = abs(arr[1][0][1] - arr[0][0][1]) # p3y - p1y 
  dx1 = abs(arr[0][0][0] - arr[0][1][0]) 
  dx2 = abs(arr[1][0][0] - arr[1][1][0]) 

  if dx1 < dx2:
    x_avg = (arr[0][0][0] + arr[0][1][0]) // 2 
    dx = dx1
  else: 
    x_avg = (arr[1][0][0] - arr[1][1][0]) // 2 
    dx = dx2


  print(x_avg, dx)

  # if(x1_avg < x2_avg):
  #   dx = abs(arr[0][0][0] - arr[0][1][0]) # p1x - p2x 
  # else: 
  #   dx = abs(arr[1][0][0] - arr[1][1][0]) # p1x - p2x 

  p1 = [x_avg - dx//4, y_avg ]
  p2 = [x_avg + dx//4, y_avg ]
  p3 = [x_avg - dx//4, y_avg + dy]
  p4 = [x_avg + dx//4, y_avg + dy]

  # for i in arr:
  #   yf = (i[0][1] + i[1][1])/2 # my y 
  #   x = [i[0][0], i[1][0]]
  #   x.sort()
  #   dx = (x[1] - x[0])/4
  #   xm = x[0] + dx
  #   xx = x[1] - dx
  #   final_pts.append([xm, yf])
  #   final_pts.append([xx, yf])
    
  # # arr.sort()

  # p1 = arr[0][0]
  # p2 = arr[0][1]
  # p3 = arr[1][0]
  # p4 = arr[1][1]

  # dx = 2

  # y1_avg = (p3[1]+p1[1]) / 2
  # x1_avg = (p3[0]+p1[0]) / 2
  # y2_avg = (p4[1]+p2[1]) / 2
  # x2_avg = (p4[0]+p2[0]) / 2
  # xf = (x1_avg + x2_avg) / 2
  # yf = (y1_avg + y2_avg) / 2
  # x1 = [xf-dx, yf-dx]
  # x2 = [xf+dx, yf-dx]
  # x3 = [xf-dx, yf+dx]
  # x4 = [xf+dx, yf+dx]

  return [p1,p2,p3,p4]
  # return final_pts

def dynamicHmg(pts):
  arr = copy.deepcopy(pts)
  arr_len = len(arr)
  if arr_len <= 1: 
    print("\n not enough lanes ", arr_len)
    return
  
  midpoint = arr_len//2

  

  print("total lanes ", arr_len)

  midpoint = 0
  y1 = [j[1] for j in arr[midpoint]] # fills array with y values 
  y2 = [j[1] for j in arr[midpoint+1]]


  # print(arr[midpoint-1])
  # print(arr[midpoint])

  # print(y1)
  # print(y2)
  total_points = find_closest(y1,y2, arr[midpoint],  arr[midpoint+1])
  rec_points = make_rect(total_points)
  # indices = []

  # for i in range(len(y1)):
  #   for j in range(len(y2)):
  #     if y1[i] == y2[j]: # if they are equal then store their index
  #       indices.append([i,j])

  # total_points = []
  # for i in (indices):
  #   t1 = arr[midpoint-1][i[0]]
  #   t2 = arr[midpoint][i[1]]
  #   total_points.append([t1, t2])

  # final_pts = []

  # for i in range(len(total_points)): # fix this | it should be acc to which is smaller 
  #   if total_points[i][1] < total_points[i][0]:
  #     final_pts.append(total_points[i][1])
  #     final_pts.append(total_points[i][0])
  #   else:
  #     final_pts.append(total_points[i][0])
  #     final_pts.append(total_points[i][1])


  final_pts = []
  final_pts.append(total_points[0][0])
  final_pts.append(total_points[0][1])
  final_pts.append(total_points[1][0])
  final_pts.append(total_points[1][1])

  final_pts = np.array(final_pts).astype(int)
  rec_points = np.array(rec_points).astype(int)

  print("total points: ", final_pts )
  print("rect points: ", rec_points )
  h, _ = cv2.findHomography(final_pts, rec_points, cv2.RANSAC, 5.0)
  print("homography -- ", h)
  return final_pts, rec_points, h

class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, cfg, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg)
        self._ipm_remap_file_path = ipm_remap_file_path

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None,
                    with_lane_fit=True, data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param with_lane_fit:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }
        if not with_lane_fit:
            tmp_mask = cv2.resize(
                mask_image,
                dsize=(source_image.shape[1], source_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            source_image = cv2.addWeighted(source_image, 0.6, tmp_mask, 0.4, 0.0, dst=source_image)
            return {
                'mask_image': mask_image,
                'fit_params': None,
                'source_image': source_image,
            }

        # lane line fit
        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        for lane_index, coords in enumerate(lane_coords):
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
            else:
                raise ValueError('Wrong data source now only support tusimple')
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

            lane_pts = []
            for index in range(0, plot_y.shape[0], 5):
                src_x = self._remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                if src_x <= 0:
                    continue
                src_y = self._remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)

        vanishing_point = find_vanishing(src_lane_pts)
        fp, rp, h_mat = dynamicHmg(src_lane_pts)
        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1]
        for index, single_lane_pts in enumerate(src_lane_pts):
            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
            if data_source == 'tusimple':
                start_plot_y = 240
                end_plot_y = 720
            else:
                raise ValueError('Wrong data source now only support tusimple')
            step = int(math.floor((end_plot_y - start_plot_y) / 10))
            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
                idx_low = np.argmax(fake_diff_smaller_than_zero)
                idx_high = np.argmin(fake_diff_bigger_than_zero)

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue

                lane_color = self._color_map[index].tolist()
                cv2.circle(source_image, (int(interpolation_src_pt_x),
                                          int(interpolation_src_pt_y)), 5, lane_color, -1)
        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
            'vanishing_point':vanishing_point,
            'homography':h_mat,
            'fp':fp,
            'rp':rp
        }

        return ret
