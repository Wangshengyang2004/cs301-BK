
# 导出模型
import torch
# 导入InceptionResnetV1模型
from facenet.FACENET import InceptionResnetV1
# 实例化
facenet = InceptionResnetV1(is_train=False,embedding_length=128,num_classes=14575)
# 从训练文件中加载
facenet.load_state_dict(torch.load('./facenet/facenet_best.pt'))
facenet.eval()
sm = torch.jit.script(facenet)
sm.save("facenet_sc.pt")