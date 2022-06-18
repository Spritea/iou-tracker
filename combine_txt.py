import os
from tqdm import tqdm
# this file combine_txt.py is to
# combine mots_seg_track txt outputs of each class to one file.

val_in_trainval_seqmap=[2,6,7,8,10,13,14,16,18]
seqmap=val_in_trainval_seqmap

file_dir="Adelaidet_result/training_dir/COCO_pretrain/CondInst_MS_R_50_1x_kitti_mots/to_mots_txt/mots_seg_track/val_in_trainval/"
for seq_one in tqdm(seqmap):
    car_file=file_dir+'car/'+str(seq_one).zfill(4)+'.txt'
    pedestrian_file=file_dir+'pedestrian/'+str(seq_one).zfill(4)+'.txt'
    file_inputs=[car_file,pedestrian_file]
    out_dir=file_dir+'both_car_pedestrian/'
    out_path=out_dir+str(seq_one).zfill(4)+'.txt'
    os.makedirs(out_dir,exist_ok=True)
    
    with open(out_path, 'w') as outfile:
        for fname in file_inputs:
            with open(fname,'r') as infile:
                for line in infile:
                    if 'pedestrian' in fname:
                        line_split=line.strip().split(' ')
                        # change track id to not start with 1,
                        # avoid being same with car track id.
                        line_split[1]=str(int(line_split[1])+1000)
                        line_new=' '.join(line_split)+'\n'
                        line=line_new
                    outfile.write(line)