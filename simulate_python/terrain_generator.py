import numpy as np
import cv2
import os
from noise import pnoise2
import xml.etree.ElementTree as ET

# ==========================================
# 配置区
# ==========================================
ROBOT = "go2_Oracle"
INPUT_SCENE_PATH = "../unitree_robots/go2_Oracle/scene.xml" 

# 输出文件的路径也可以根据你的习惯改，比如输出到当前目录
OUTPUT_SCENE_PATH = f"../unitree_robots/{ROBOT}/scene_terrain1.xml"

class TerrainGenerator:
    def __init__(self, input_xml):
        self.tree = ET.parse(input_xml)
        self.root = self.tree.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")
        if self.asset is None:
            self.asset = ET.SubElement(self.root, "asset")

    def save(self, output_xml):
        self.tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        print(f"[INFO] 地形已生成并保存至: {output_xml}")

    def AddBox(self, position=[1.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], size=[0.1, 0.1, 0.1]):
        ET.SubElement(self.worldbody, "geom", {
            "type": "box",
            "pos": f"{position[0]} {position[1]} {position[2]}",
            "euler": f"{euler[0]} {euler[1]} {euler[2]}",
            "size": f"{size[0]} {size[1]} {size[2]}",
            "rgba": "0.5 0.5 0.5 1"
        })

    def AddRoughGround(self, init_pos=[1.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], nums=[10, 10], 
                       box_size=[0.5, 0.5, 0.5], box_euler=[0.0, 0.0, 0.0], separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05], box_euler_rand=[0.2, 0.2, 0.2], 
                       separation_rand=[0.05, 0.05]):
        """通过随机排列立方体实现杂乱地形"""
        for i in range(nums[0]):
            for j in range(nums[1]):
                # 计算基础位置
                curr_x = init_pos[0] + i * (box_size[0] * 2 + separation[0]) + np.random.uniform(-separation_rand[0], separation_rand[0])
                curr_y = init_pos[1] + j * (box_size[1] * 2 + separation[1]) + np.random.uniform(-separation_rand[1], separation_rand[1])
                curr_z = init_pos[2]
                
                # 随机增量
                s_rand = np.random.uniform(-1, 1, 3) * box_size_rand
                e_rand = np.random.uniform(-1, 1, 3) * box_euler_rand
                
                self.AddBox(
                    position=[curr_x, curr_y, curr_z],
                    euler=np.array(box_euler) + e_rand,
                    size=np.array(box_size) + s_rand
                )

    def AddPerlinHeighField(self, position=[0.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], size=[2.0, 2.0], 
                           height_scale=0.2, negative_height=0.2, image_width=128, img_height=128,
                           smooth=50.0, perlin_octaves=6, perlin_persistence=0.5, perlin_lacunarity=2.0,
                           output_hfield_image="height_field.png"):
        """基于 Perlin 噪声生成高程图"""
        # 生成噪声图
        img = np.zeros((img_height, image_width), dtype=np.float32)
        for y in range(img_height):
            for x in range(image_width):
                img[y][x] = pnoise2(x / smooth, y / smooth, 
                                    octaves=perlin_octaves, 
                                    persistence=perlin_persistence, 
                                    lacunarity=perlin_lacunarity, 
                                    repeatx=image_width, repeaty=img_height, base=0)
        
        # 归一化到 0-255 并保存
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        cv2.imwrite(output_hfield_image, img)
        
        # 在 XML 中添加 asset 和 geom
        hfield_name = "perlin_hfield"
        ET.SubElement(self.asset, "hfield", {
            "name": hfield_name,
            "nrow": str(img_height),
            "ncol": str(image_width),
            "size": f"{size[0]} {size[1]} {height_scale} {negative_height}",
            "file": output_hfield_image
        })
        ET.SubElement(self.worldbody, "geom", {
            "type": "hfield",
            "hfield": hfield_name,
            "pos": f"{position[0]} {position[1]} {position[2]}",
            "euler": f"{euler[0]} {euler[1]} {euler[2]}",
            "rgba": "0.4 0.3 0.2 1"
        })

    def AddStairs(self, init_pos=[1.0, 0.0, 0.0], yaw=0.0, width=0.2, height=0.15, length=1.5, stair_nums=10):
        """添加楼梯"""
        for i in range(stair_nums):
            pos = [
                init_pos[0] + i * width * np.cos(yaw),
                init_pos[1] + i * width * np.sin(yaw),
                init_pos[2] + i * height
            ]
            self.AddBox(position=pos, euler=[0, 0, yaw], size=[width/2, length/2, height/2 * (i+1)])

# ==========================================
# 3. 运行生成逻辑
# ==========================================
if __name__ == "__main__":
    gen = TerrainGenerator(INPUT_SCENE_PATH)

    # 示例 1：在前方生成一段楼梯
    gen.AddStairs(init_pos=[1.5, 0.0, 0.05], stair_nums=8, width=0.25, height=0.05)

    # 示例 2：在远处生成一片 Perlin 噪声崎岖地面
    gen.AddPerlinHeighField(position=[5.0, 0.0, 0.0], size=[5.0, 5.0], height_scale=0.3)

    # 示例 3：在侧面生成杂乱的方块阵（乱石阵）
    gen.AddRoughGround(init_pos=[0.0, 3.0, 0.0], nums=[5, 5], box_size=[0.2, 0.2, 0.1])

    gen.save(OUTPUT_SCENE_PATH)