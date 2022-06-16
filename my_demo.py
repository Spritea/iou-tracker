import argparse

from iou_tracker import track_iou
from util import load_mot, save_to_csv
import os
from tqdm import tqdm

def main(args):
  
    def process_seq_one(det_path,out_path):
        detections = load_mot(det_path, nms_overlap_thresh=None, with_classes=False)
        tracks = track_iou(detections, args.sigma_l, args.sigma_h, args.sigma_iou, args.t_min)
        save_to_csv(out_path, tracks, fmt=args.format)
    
    # car:1,pedestrian:2.
    categories=["car","pedestrian"]
    # train_in_trainval_seqmap=[0,1,3,4,5,9,11,12,15,17,19,20]
    val_in_trainval_seqmap=[2,6,7,8,10,13,14,16,18]
    seqmap=val_in_trainval_seqmap
    
    det_dir='Adelaidet_result/training_dir/COCO_pretrain/FCOS_R_50_1x_kitti_mots/to_mots_txt/mot_det/val_in_trainval/'
    out_dir='Adelaidet_result/training_dir/COCO_pretrain/FCOS_R_50_1x_kitti_mots/to_mots_txt/mot_det_track/val_in_trainval/'
    os.makedirs(out_dir,exist_ok=True)
    
    for cat in tqdm(categories):
        for seq_one in seqmap:
            det_path=det_dir+cat+'/'+str(seq_one).zfill(4)+'.txt'
            out_path=out_dir+cat+'/'+str(seq_one).zfill(4)+'.txt'
            process_seq_one(det_path,out_path)
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IOU/V-IOU Tracker demo script")
    parser.add_argument('-v', '--visual', type=str, help="visual tracker for V-IOU. Currently supported are "
                                                         "[BOOSTING, MIL, KCF, KCF2, TLD, MEDIANFLOW, GOTURN, NONE] "
                                                         "see README.md for furthert details")
    parser.add_argument('-hr', '--keep_upper_height_ratio', type=float, default=1.0,
                        help="Ratio of height of the object to track to the total height of the object "
                             "for visual tracking. e.g. upper 30%%")
    parser.add_argument('-f', '--frames_path', type=str,
                        help="sequence frames with format '/path/to/frames/frame_{:04d}.jpg' where '{:04d}' will "
                             "be replaced with the frame id. (zero_padded to 4 digits, use {:05d} for 5 etc.)")
    # parser.add_argument('-d', '--detection_path', type=str, required=True,
    #                     help="full path to CSV file containing the detections")
    # parser.add_argument('-o', '--output_path', type=str, required=True,
    #                     help="output path to store the tracking results "
    #                          "(MOT challenge/Visdrone devkit compatible format)")
    parser.add_argument('-sl', '--sigma_l', type=float, default=0,
                        help="low detection threshold")
    parser.add_argument('-sh', '--sigma_h', type=float, default=0.5,
                        help="high detection threshold")
    parser.add_argument('-si', '--sigma_iou', type=float, default=0.5,
                        help="intersection-over-union threshold")
    parser.add_argument('-tm', '--t_min', type=float, default=2,
                        help="minimum track length")
    parser.add_argument('-ttl', '--ttl', type=int, default=1,
                        help="time to live parameter for v-iou")
    parser.add_argument('-nms', '--nms', type=float, default=None,
                        help="nms for loading multi-class detections")
    parser.add_argument('-fmt', '--format', type=str, default='motchallenge',
                        help='format of the detections [motchallenge, visdrone]')

    args = parser.parse_args()
    assert not args.visual or args.visual and args.frames_path, "visual tracking requires video frames, " \
                                                                "please specify via --frames_path"

    assert 0.0 < args.keep_upper_height_ratio <= 1.0, "only values between 0 and 1 are allowed"
    assert args.nms is None or 0.0 <= args.nms <= 1.0, "only values between 0 and 1 are allowed"
    main(args)
