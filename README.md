# Tianchi_meinianAI

����˵����

��Ϊ�����������֣�
1.���ݴ���data_pro.py��:
	A.����ȥ�أ�ת���ɱ�׼��ʽ��ÿ��Ϊ'vid'����Ϊ'table_id'����[data_pro0]
	B.label����  [data_pro1]

2.������ȡ��feat_ext.py��:
	A.ͨ��ӳ����ϴ���ݣ���ȡ��ֵ������[Extract_feat0,Extract_feat1,Extract_feat2]
	B.ͨ����ȡ�ؼ��֣���ȡ�ı�������[Extract_feat3]
	C.�����ϲ�����ȡ��ѵ�������Ԥ�����[Feature_concat]

3.lgb��ģ(main.py)
	A.����ҪԤ���5��label�ֱ�ģ��lgb��ģ��
	B.����ͨ��GridSearch���[lgb_CV.py]��main�����в����д��


ע��main�����еĸ����м䲽�趼�б����������һ������ֹ���ɴ�main������ѡȡ��Ӧ����������С�



package��
	numpy 1.14.0
	pandas 0.22.0
	scikit-learn 0.19.1
	ligthgbm 2.0.12