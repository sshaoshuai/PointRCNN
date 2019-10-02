# Notes for modify PointRCNN

os.path.basename    返回path最后的文件名, 配合split使用效果更佳

和nn.Module不同，调用tensor.cuda()只是返回这个tensor对象在GPU内存上的拷贝，而不会对自身进行改变。
因此必须对tensor进行重新赋值，即tensor=tensor.cuda().

分析dataset数据的准备工作:

kittiRCNN:root_dir, npoints=16384, split='train', classes='Car', mode='TRAIN', random_select=True,
                 logger=None, rcnn_training_roi_dir=None, rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None,
                 rcnn_eval_feature_dir=None, gt_database_dir=None
                
                
不同点在于:
    1. fat需要360°输入
    2. 而且对于目标任务 是多个种类都需要进行训练
    
    
修改策略:
    1. 改输入结构