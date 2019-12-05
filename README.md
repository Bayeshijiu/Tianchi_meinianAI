# Tianchi_meinianAI

程序说明：

分为以下三个部分: 


1.数据处理（data_pro.py）: 

	A.数据去重，转换成标准格式（每行为'vid'，列为'table_id'）。[data_pro0] 

	B.label清理  [data_pro1] 

2.特征提取（feat_ext.py）:

	A.通过映射清洗数据，提取数值特征。[Extract_feat0,Extract_feat1,Extract_feat2] 

	B.通过提取关键字，提取文本特征。[Extract_feat3] 

	C.特征合并，提取出训练矩阵和预测矩阵。[Feature_concat] 

3.lgb建模(main.py)

	A.对需要预测的5列label分别单模型lgb建模。 

	B.参数通过GridSearch获得[lgb_CV.py]，main函数中不进行此项。 


注：main函数中的各个中间步骤都有保存输出，万一程序终止，可从main函数中选取对应步骤继续进行。



package：

	numpy 1.14.0
	pandas 0.22.0
	scikit-learn 0.19.1
	ligthgbm 2.0.12
