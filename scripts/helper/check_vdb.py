# 纯测试：无torch、无深度图，绝对合法数据
import numpy as np
import vdbfusion

# 初始化
vol = vdbfusion.VDBVolume(0.05, 0.15)
# 合法随机点云
pts = np.random.rand(1000,3).astype(np.float64)
#pose = np.eye(4).astype(np.float64)

#vol.integrate(pts, pose)

position = np.array([0.0, 0.0, 0.0]).astype(np.float64)

vol.integrate(pts, position)

print("✅ 测试成功！VDBFusion 无问题")
