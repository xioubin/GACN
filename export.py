'''export model to onnx'''

from nets.gacn_net import GACN_Fuse

gacn = GACN_Fuse()

gacn.export()