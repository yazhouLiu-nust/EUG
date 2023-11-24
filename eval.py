import numpy as np
from scipy.interpolate import interp1d
import os
import cv2 as cv

class AnomalyEvaluator():
    def __init__(self,ignor, road):
        self.ignore_label = ignor
        self.quantization = 1000
        self.anomaly_range = [0, 1]
        self.cmat = np.zeros(shape=[self.quantization, 2, 2])  
        self.road_id = road
    
    def create_roi(self, gt_label):
        roi = (gt_label != self.ignore_label).astype(np.bool_)
        # TODO limit roi to regions around road, so the metric is focused on anomalies on the road
        return roi

    def compute_cmat(self, gt_label, prob):
        roi = self.create_roi(gt_label)
        prob = prob[roi]
        area = prob.__len__()
        gt_label = gt_label[roi]

        gt_mask_road = (gt_label == self.road_id)
        gt_mask_obj = ~gt_mask_road

        gt_area_true = np.count_nonzero(gt_mask_obj)
        gt_area_false = area - gt_area_true

        prob_at_true = prob[gt_mask_obj]
        prob_at_false = prob[~gt_mask_obj]

        tp, _ = np.histogram(prob_at_true, self.quantization, range=self.anomaly_range)
        tp = np.cumsum(tp[::-1])

        fn = gt_area_true - tp

        fp, _ = np.histogram(prob_at_false, self.quantization, range=self.anomaly_range)
        fp = np.cumsum(fp[::-1])

        tn = gt_area_false - fp

        cmat = np.array([
            [tp, fp],
            [fn, tn],
            ]).transpose(2, 0, 1)
        
        if area > 0:
            cmat = cmat.astype(np.float64) / area
        else:
            cmat[:] = 0

        if np.any((cmat>1) | (cmat<0)):
            assert False, "Error in computing tp,fp,fn,tn. Some values larger than 1 or less than 0 {}".format(cmat)

        return cmat, area > 0

    def add_batch(self, gt_image_t, output):
        gt_image = gt_image_t.cpu().numpy()
        pre_image = output["anomaly_score"].cpu().numpy()[:, 0, ...]
        assert gt_image.shape == pre_image.shape
        for b in range(0, pre_image.shape[0]):
            cmat_b, valid_frame = self.compute_cmat(gt_image[b, ...], pre_image[b, ...])
            if valid_frame:
                self.cmat += cmat_b
                
    def add_batch_lst(self, gt_lst, output_lst):
        
        for idx,gt in enumerate(gt_lst):
            im = output_lst[idx]        
            
            assert im.shape == gt.shape
            cmat_b, valid_frame = self.compute_cmat(gt, im)
            if valid_frame:
                self.cmat += cmat_b

    def reset(self):
        self.cmat[:] = 0

    def compute_stats(self):
        tp = self.cmat[:, 0, 0]
        fp = self.cmat[:, 0, 1]
        fn = self.cmat[:, 1, 0]
        tn = self.cmat[:, 1, 1]

        tp_rates = tp / (tp+fn) # = recall
        fp_rates = fp / (fp+tn)

        fp[(tp+fp) == 0] = 1e-9
        precision = tp / (tp+fp) 

        area_under_TPRFPR = np.trapz(tp_rates, fp_rates)
        AP = np.trapz(precision, tp_rates)

        f = interp1d(tp_rates, fp_rates, kind="linear")
        FPRat95 = f(0.95)
        
        return AP, FPRat95
        
  
def file_filter(f):
    if f[-4:] in ['.png']:
        return True
    else:
        return False
    
def file_finder(in_dir):
    
    files = os.listdir(in_dir)
    
    files = list(filter(file_filter,files))
    return files

        
def load_data(gt_dir, ot_dir, rp_str):
    
    img_files=file_finder(gt_dir)
    
    gt_lst=[]
    pr_lst=[]
    
    for idx, file in enumerate(img_files):
        #..label img
        full_path = os.path.join(gt_dir,file)
        lab=cv.imread(full_path,-1)
        
        #..pred img
        name_score = file.replace(rp_str[0], rp_str[1])
        path_score = os.path.join(ot_dir,name_score)
        
        img=cv.imread(path_score, -1)
        
        if img.shape[0]!=lab.shape[0]:
            img_r = cv.resize(img,(lab.shape[1],lab.shape[0]),interpolation = cv.INTER_CUBIC)
        else:
            img_r = img
        
        #to float
        img_f=img_r.astype('float32')/255.
        
        pr_lst.append(img_f)
        gt_lst.append(lab)
    return gt_lst,pr_lst   
     


def load_gt_and_pred():
    #..the output path
    #output_path='G:\\segData\\PubCode\\output.ensemble'
    #output_path='G:\\segData\\PubCode\\output.EUG_tiny'
    #output_path='G:\\segData\\PubCode\\output.EUG_base'
    output_path='G:\\segData\\PubCode\\output.EUG_heter'
    #output_path='f:\\segData\\mskRes\\results'
    
    #..the gt path
    gt_path = 'G:\\segData\\PubCode\\GTs'
    #gt_path = 'f:\\segData\\GTs'
    
    #..the path of dataset gts
    gt_path_Laf = os.path.join(gt_path,'LaF')
    gt_path_RA = os.path.join(gt_path,'RA')
    gt_path_RO = os.path.join(gt_path,'RO')
    
    #..the path of dataset outputs
    ot_path_Laf = os.path.join(output_path,'LaF')
    ot_path_RA = os.path.join(output_path,'RA')
    ot_path_RO = os.path.join(output_path,'RO')
    
    
    print(output_path)
    
    lf_str=['_gtCoarse_labelTrainIds.png', '_leftImg8bitgrey.jpg']
    ra_str=['.png', 'grey.jpg']
    ro_str=['_labels_semantic.png', 'grey.jpg']
    
    #lf_str=['_gtCoarse_labelTrainIds.png', '_leftImg8bit.jpg']
    #ra_str=['.png', '.jpg']
    #ro_str=['_labels_semantic.png', '.jpg']
    
    
    #..ro
    gt_lst_ro,pr_lst_ro=load_data(gt_path_RO, ot_path_RO, ro_str)
    ev_ro = AnomalyEvaluator(255,0)
    ev_ro.add_batch_lst(gt_lst_ro,pr_lst_ro)
    AP, FPR95=ev_ro.compute_stats()
    print("RO %d: AP:FPR95   %f,%f"%(len(gt_lst_ro),AP*100,FPR95*100))
    
    #..ra
    gt_lst_ra,pr_lst_ra=load_data(gt_path_RA, ot_path_RA, ra_str)
    ev_ra = AnomalyEvaluator(255,1)
    ev_ra.add_batch_lst(gt_lst_ra,pr_lst_ra)
    AP, FPR95=ev_ra.compute_stats()
    print("RA %d: AP:FPR95   %f,%f"%(len(gt_lst_ra),AP*100,FPR95*100))
    
    #..lf
    gt_lst_lf,pr_lst_lf=load_data(gt_path_Laf, ot_path_Laf, lf_str)
    ev_lf = AnomalyEvaluator(255,1)
    ev_lf.add_batch_lst(gt_lst_lf,pr_lst_lf)
    AP, FPR95=ev_lf.compute_stats()
    print("LF %d: AP:FPR95   %f,%f"%(len(gt_lst_lf),AP*100,FPR95*100))
    
    
    
#def doEvaluation():
load_gt_and_pred()    
