import numpy as np
import csv
import os
from util import iou
from itertools import compress

# util things by cws.

def load_mots_det_seg(detections, nms_overlap_thresh=None, with_classes=False, nms_per_class=False):
    # load mots_det_seg
    if nms_overlap_thresh:
        assert with_classes, "currently only works with classes available"

    # data = []
    # if type(detections) is str:
    #     raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
    #     if np.isnan(raw).all():
    #         raw = np.genfromtxt(detections, delimiter=' ', dtype=np.float32)

    # else:
    #     # assume it is an array
    #     assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
    #     raw = detections.astype(np.float32)

    # end_frame = int(np.max(raw[:, 0]))
    # # for i in range(1, end_frame+1):
    # # need to start from 0, for KITTI MOTS.
    # # the start from 1 is for MOT17-challenge only.
    # for i in range(0, end_frame+1):
    #     idx = raw[:, 0] == i
    #     bbox = raw[idx, 2:6]
    #     bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
    #     # bbox -= 1  # correct 1,1 matlab offset
    #     scores = raw[idx, 6]
        
    #     # the with_class choise part is removed.
    #     classes = ['pedestrian']*bbox.shape[0]

    #     dets = []
    #     for bb, s, c in zip(bbox, scores, classes):
    #         dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c})
    #     data.append(dets)

    with open(detections,'r') as f:
        raw_string=f.readlines()
    raw_no_end_split=[x.strip().split(',') for x in raw_string]
    # get seg part.
    seg_list=[]
    det_only_list=[]
    for item in raw_no_end_split:
        seg_one=item[-3:]
        det_one=[float(x) for x in item[:-3]]
        seg_list.append(seg_one)
        det_only_list.append(det_one)
    det_np=np.array(det_only_list,dtype=np.float32)
    
    my_data=[]
    end_frame = int(np.max(det_np[:, 0]))
    # for i in range(1, end_frame+1):
    # need to start from 0, for KITTI MOTS.
    # the start from 1 is for MOT17-challenge only.
    for i in range(0, end_frame+1):
        idx = det_np[:, 0] == i
        bbox = det_np[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        # no use this, since would get -1 result.
        # it's ok to use, since save_to_csv add 1 back to topleft_x,topleft_y.
        bbox -= 1  # correct 1,1 matlab offset
        scores = det_np[idx, 6]
        
        # the with_class choise part is removed.
        classes = ['pedestrian']*bbox.shape[0]

        # below append seg part to each line.
        seg_list_frame_one=list(compress(seg_list,idx))
        my_dets = []
        for bb, s, c,seg in zip(bbox, scores, classes,seg_list_frame_one):
            my_dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c,'seg': seg})
        my_data.append(my_dets)

    return my_data


def track_iou_det_seg(detections, sigma_l, sigma_h, sigma_iou, t_min):
    # save to txt file, seperated by space, to be compatible with KITTI MOTS.

    tracks_active = []
    tracks_finished = []

    # for frame_num, detections_frame in enumerate(detections, start=1):
    # note here frame_num should start from 0 in KITTI MOTS txt.
    for frame_num, detections_frame in enumerate(detections):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])
                    # add seg part to track.
                    track['seg_all'].append(best_match['seg'])

                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        # new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        # add seg part to new track.
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num,'seg_all':[det['seg']]} for det in dets]

        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished

def save_to_txt(out_path, tracks,category_id):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as  f:
        id_ = 1
        for track in tracks:
            for i, seg_one in enumerate(track['seg_all']):
                frame=track['start_frame'] + i
                track_id=id_
                obj_class=category_id
                # seg_one is already string.
                img_h=seg_one[0]
                img_w=seg_one[1]
                rle=seg_one[2]
                line=[str(frame),str(track_id),str(obj_class),img_h,img_w,rle]
                # kitti mots need ' ', not ',' in txt file.
                line=' '.join(line)+'\n'
                f.write(line)
            id_ += 1

