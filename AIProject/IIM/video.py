from scipy.spatial import distance_matrix
import cv2
import os
import sys
import numpy as np
from scipy import spatial as ss
from misc.utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter


# def read_pred_and_gt(pred_file, gt_file):
#     pred_data = []
#     with open(pred_file, 'r') as f:
#         for line in f:
#             if line.strip() == '':
#                 continue
#             items = line.strip().split()
#             frame_idx = int(items[0])
#             num_points = int(items[1])
#             points = np.zeros((num_points, 2))
#             for i in range(num_points):
#                 points[i, 0] = float(items[2 + i * 2])
#                 points[i, 1] = float(items[2 + i * 2 + 1])
#             pred_data.append({'num': num_points, 'points': points})

#     gt_data = []
#     with open(gt_file, 'r') as f:
#         for line in f:
#             if line.strip() == '':
#                 continue
#             items = line.strip().split()
#             frame_idx = int(items[0])
#             num_points = int(items[1])
#             points = np.zeros((num_points, 2))
#             for i in range(num_points):
#                 points[i, 0] = float(items[2 + i * 3])
#                 points[i, 1] = float(items[2 + i * 3 + 1])
#             sigma = np.zeros((num_points, 2))
#             for i in range(num_points):
#                 sigma[i, 0] = float(items[2 + i * 3 + 2])
#                 sigma[i, 1] = sigma[i, 0] * 2
#             level = int(items[-1])
#             gt_data.append({'num': num_points, 'points': points, 'sigma': sigma, 'level': level})

#     return pred_data, gt_data




def main():
    dataset = 'JHU'
    dataRoot = '../ProcessedData/' + dataset
    gt_file = dataRoot + '/val_gt_loc.txt'
    img_path = dataRoot + '/videos'
    pred_file = './saved_exp_results/JHU_VGG16_FPN_val.txt'
    exp_name = './saved_exp_results/videos_vis_results'

    
    # Open input video
    input_video = cv2.VideoCapture(dataRoot+'/videos/video1.mp4')
    fps = input_video.get(cv2.CAP_PROP_FPS)
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    
    # Get input video properties
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create output video object
    output_video = cv2.VideoWriter(exp_name + '.mp4', fourcc, fps, (width, height))

    # Load predicted bounding box information
    with open(pred_file, 'r') as f:
        pred_lines = f.readlines()

    # Loop over frames of input video
    frame_num = 0
    while True:
        # Read next frame from input video
        ret, frame = input_video.read()

        # If frame could not be read, break loop
        if not ret:
            break

        # Increment frame number
        frame_num += 1

        # Get predicted bounding box for current frame
        pred_line = pred_lines[frame_num - 1]
        pred_parts = pred_line.strip().split()
        xmin, ymin, xmax, ymax, score = [int(float(x)) for x in pred_parts[:5]]

        # Draw bounding box on frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Write frame to output video
        output_video.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release input and output videos
    input_video.release()
    output_video.release()

    # Close all windows
    cv2.destroyAllWindows()


    pred_data, gt_data = read_pred_and_gt(pred_file,gt_file)


    for i in range(total_frames):
        ret, frame = input_video.read()
        if not ret:
            break
        
        gt_p,pred_p,fn_gt_index,tp_pred_index,fp_pred_index,ap,ar= [],[],[],[],[],[],[]
        
        if gt_data[i]['num'] ==0 and pred_data[i]['num'] !=0:
            pred_p =  pred_data[i]['points']
            fp_pred_index = np.array(range(pred_p.shape[0]))
            ap = 0
            ar = 0

        if pred_data[i]['num'] ==0 and gt_data[i]['num'] !=0:
            gt_p = gt_data[i]['points']
            fn_gt_index = np.array(range(gt_p.shape[0]))
            sigma_l = gt_data[i]['sigma'][:,1]
            ap = 0
            ar = 0

        if gt_data[i]['num'] !=0 and pred_data[i]['num'] !=0:
            pred_p =  pred_data[i]['points']    
            gt_p = gt_data[i]['points']
            sigma_l = gt_data[i]['sigma'][:,1]
            level = gt_data[i]['level']  
            
            # dist
            dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
            match_matrix = np.zeros(dist_matrix.shape,dtype=bool)
            for i_pred_p in range(pred_p.shape[0]):
                pred_dist = dist_matrix[i_pred_p,:]
                match_matrix[i_pred_p,:] = pred_dist<=sigma_l
                
            # hungarian outputs a match result, which may be not optimal. 
            # Nevertheless, the number of tp, fp, tn, fn are same under different match results
            # If you need the optimal result for visualzation, 
            # you may treat it as maximum flow problem. 
            tp, assign = hungarian(match_matrix)
            fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
            tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
            tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
            fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]



            pre = tp_pred_index.shape[0]/(tp_pred_index.shape[0]+fp_pred_index.shape[0]+1e-20)
            rec = tp_pred_index.shape[0]/(tp_pred_index.shape[0]+fn_gt_index.shape[0]+1e-20)
            print(pre, rec)

        img = cv2.imread(img_path + '/' + str(i_sample) + '.jpg')#bgr
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        
        point_r_value = 5
        thickness = 3
        if gt_data[i_sample]['num'] !=0:
            for i in range(gt_p.shape[0]):
                if i in fn_gt_index:                
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),point_r_value,(0,0,255),-1)# fn: red
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),sigma_l[i],(0,0,255),thickness)#  
                else:
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),sigma_l[i],(0,255,0),thickness)# gt: green
        if pred_data[i_sample]['num'] !=0:
            for i in range(pred_p.shape[0]):
                if i in tp_pred_index:
                    cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value,(0,255,0),-1)# tp: green
                else:                
                    cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value*2,(255,0,255),-1) # fp: Magenta

        cv2.imwrite(exp_name+'/'+str(i_sample)+ '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6] + '.jpg', img)


# dataset = 'JHU'
# dataRoot = '../ProcessedData/' + dataset
# gt_file = dataRoot + '/val_gt_loc.txt'
# img_path = dataRoot + '/videos'
# pred_file = './saved_exp_results/JHU_VGG16_FPN_val.txt'
# exp_name = './saved_exp_results/videos_vis_results'

# cap = cv2.VideoCapture('../ProcessedData/JHU/videos/video1.mp4')
# fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(exp_name+'/output_video.mp4', fourcc, fps, (640, 480))


# # Hungarian algorithm for solving matching problem
# def hungarian(match_matrix):
#     from scipy.optimize import linear_sum_assignment
#     tp_gt_index = []
#     tp_pred_index = []
#     for i in range(match_matrix.shape[0]):
#         if match_matrix[i,:].sum() > 0:
#             tp_pred_index.append(i)
#             tp_gt_index.append(np.where(match_matrix[i,:])[0][0])
#     tp = len(tp_gt_index)
#     fp_pred_index = list(set(range(match_matrix.shape[0]))-set(tp_pred_index))
#     fn_gt_index = list(set(range(match_matrix.shape[1]))-set(tp_gt_index))
#     return tp, np.array(tp_gt_index)

# # Read predicted and ground truth data from files
# def read_pred_and_gt(pred_file, gt_file):
#     pred_data = {}
#     gt_data = {}
#     with open(pred_file, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip().split(',')
#             print(int(line[0]))
#             frame_num = int(line[0])
#             num_points = int(line[1])
#             print(num_points)
#             points = []
#             for i in range(2, len(line)):
#                 point = [int(float(p)) for p in line[i].split()]
#                 points.append(point)
#             pred_data[frame_num] = {'num': num_points, 'points': np.array(points)}

#     with open(gt_file, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip().split(',')
#             frame_num = int(line[0])
#             num_points = int(line[1])
#             points = []
#             sigma = []
#             level = []
#             for i in range(2, len(line), 3):
#                 point = [int(float(p)) for p in line[i:i+2]]
#                 sigma_val = float(line[i+2])
#                 level_val = int(i/3)+1
#                 points.append(point)
#                 sigma.append([sigma_val, sigma_val])
#                 level.append(level_val)
#             gt_data[frame_num] = {'num': num_points, 'points': np.array(points),
#                                   'sigma': np.array(sigma), 'level': np.array(level)}
#     return pred_data, gt_data


# def main():
#     pre = 0
#     rec = 0
#     pred_data, gt_data = read_pred_and_gt(pred_file, gt_file)

#     for i in range(total_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gt_p,pred_p,fn_gt_index,tp_pred_index,fp_pred_index,ap,ar= [],[],[],[],[],[],[]

#     if gt_data[i]['num'] ==0 and pred_data[i]['num'] !=0:
#         pred_p =  pred_data[i]['points']
#         fp_pred_index = np.array(range(pred_p.shape[0]))
#         ap = 0
#         ar = 0

#     if pred_data[i]['num'] ==0 and gt_data[i]['num'] !=0:
#         gt_p = gt_data[i]['points']
#         fn_gt_index = np.array(range(gt_p.shape[0]))
#         sigma_l = gt_data[i]['sigma'][:,1]
#         ap = 0
#         ar = 0

#     if gt_data[i]['num'] !=0 and pred_data[i]['num'] !=0:
#         pred_p =  pred_data[i]['points']    
#         gt_p = gt_data[i]['points']
#         sigma_l = gt_data[i]['sigma'][:,1]
#         level = gt_data[i]['level']  
        
#         # dist
#         dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
#         match_matrix = np.zeros(dist_matrix.shape,dtype=bool)
#         for i_pred_p in range(pred_p.shape[0]):
#             pred_dist = dist_matrix[i_pred_p,:]
#             match_matrix[i_pred_p,:] = pred_dist<=sigma_l
            
#         # hungarian outputs a match result, which may be not optimal. 
#         # Nevertheless, the number of tp, fp, tn, fn are same under different match results
#         # If you need the optimal result for visualzation, 
#         # you may treat it as maximum flow problem. 
#         tp, assign = hungarian(match_matrix)
#         fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
#         tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
#         tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
#         fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]

#         pre = tp_pred_index.shape[0]/(tp_pred_index.shape[0]+fp_pred_index.shape[0]+1e-20)
#         rec = tp_pred_index.shape[0]/(tp_pred_index.shape[0]+fn_gt_index.shape[0]+1e-20)
#         print(pre, rec)

#     img = cv2.imread(img_path + '/' + str(i) + '.jpg')#bgr
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    
#     point_r_value = 5
#     thickness = 3
#     num_gt_circles =0
#     if gt_data[i]['num'] !=0:
#         for j in range(gt_p.shape[0]):
#             if j in fn_gt_index:                
#                 cv2.circle(img,(gt_p[j][0],gt_p[j][1]),point_r_value,(0,0,255),-1)# fn: red
#                 cv2.circle(img,(gt_p[j][0],gt_p[j][1]),sigma_l[j],(0,0,255),thickness)#  
#             else:
#                 cv2.circle(img,(gt_p[j][0],gt_p[j][1]),sigma_l[j],(0,255,0),thickness)# gt: green
#             cv2.circle(img,(gt_p[i][0],gt_p[i][1]),sigma_l[i],(0,255,0),thickness)# gt: green
#             num_gt_circles += 1
#         cv2.putText(img, "Counting People: {}".format(num_gt_circles), (int(img_width*0.5), int(img_height*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # 이미지에 동그라미 수 텍스트 추가
#     if pred_data[i]['num'] !=0:
#         for j in range(pred_p.shape[0]):
#             if j in tp_pred_index:
#                 cv2.circle(img,((int(pred_p[j, 0]), int(pred_p[j, 1])), 4, (0, 255, 0), -1))
#     else:
#         cv2.circle(img, (int(pred_p[j, 0]), int(pred_p[j, 1])), 4, (0, 0, 255), -1)
#     for j in range(gt_p.shape[0]):
#         if j in fn_gt_index:
#             cv2.circle(img, (int(gt_p[j, 0]), int(gt_p[j, 1])), 4, (0, 0, 255), -1)
#         else:
#             cv2.circle(img, (int(gt_p[j, 0]), int(gt_p[j, 1])), 4, (255, 0, 0), -1)




# if __name__ == '__main__':
#     read_pred_and_gt(pred_file, gt_file)
#     main()

