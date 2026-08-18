import cv2
import numpy as np
import os
import random
from PIL import Image
import imutils
from data.util.synthetic import online_add_degradation_v2

class Photoshop(object):
    """
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """
    def __init__(self, spot_flag, scratch_flag, flwa_flag, phase):
        self.spot_flag = spot_flag
        self.scratch_flag = scratch_flag
        self.flwa_flag = flwa_flag
        self.phase = phase

    def soft_light_blend(self, src1, src2, opacity=0.5):
        # Ensure the images have the same size and type
        if src1.shape != src2.shape:
            raise ValueError("Input images must have the same dimensions and type")
        # Convert images to float32 for precision
        src1 = src1.astype(np.float32) / 255.0
        src2 = src2.astype(np.float32) / 255.0

        # Apply soft light blend formula
        blend = np.where(
            src2 < 0.5,
            2 * src1 * src2 + src1 ** 2 * (1 - 2 * src2),
            2 * src1 * (1 - src2) + np.sqrt(src1) * (2 * src2 - 1)
        )

        # Mix with original image based on opacity
        output = (1 - opacity) * src1 + opacity * blend

        # Convert back to 8-bit image
        output = (output * 255).astype(np.uint8)
        return output
    def brighten(self, src1, src2, opacity=0.5):
        if src1.shape[:2] != src2.shape[:2]:
            raise ValueError("Source image and mask must have the same dimensions")

        src1 = src1.astype(np.float32) / 255.0
        src2 = src2.astype(np.float32) / 255.0

        brightened = np.clip(src1 + src2, 0, 1)

        output = (1 - opacity) * src1 + opacity * brightened

        output = (output * 255).astype(np.uint8)
        return output
    def lighten(self, src1, src2, opacity=0.5):
        # Ensure the images have the same size and type
        if src1.shape != src2.shape:
            raise ValueError("Input images must have the same dimensions and type")

        # Convert images to float32 for precision
        src1 = src1.astype(np.float32) / 255.0
        src2 = src2.astype(np.float32) / 255.0

        # Apply lighten blend formula
        lightened = np.maximum(src1, src2)

        # Mix with original image based on opacity
        output = (1 - opacity) * src1 + opacity * lightened

        # Convert back to 8-bit image
        output = (output * 255).astype(np.uint8)
        return output
    # 滤色模式
    def screen_blend(self, src1, src2, opacity=0.5):
        # Ensure the images have the same size and type
        if src1.shape != src2.shape:
            raise ValueError("Input images must have the same dimensions and type")

        # Convert images to float32 for precision
        src1 = src1.astype(np.float32) / 255.0
        src2 = src2.astype(np.float32) / 255.0

        # Apply screen blend formula
        blend = 1 - (1 - src1) * (1 - src2)

        # Mix with original image based on opacity
        output = (1 - opacity) * src1 + opacity * blend

        # Convert back to 8-bit image
        output = (output * 255).astype(np.uint8)
        return output
    # 叠加
    def overlay_blend(self, src1, src2, opacity=0.5):
        # Ensure the images have the same size and type
        if src1.shape != src2.shape:
            raise ValueError("Input images must have the same dimensions and type")
        # Convert images to float32 for precision
        src1 = src1.astype(np.float32) / 255.0
        src2 = src2.astype(np.float32) / 255.0
        # Apply overlay blend formula
        blend = np.where(
            src1 < 0.5,
            2 * src1 * src2,
            1 - 2 * (1 - src1) * (1 - src2)
        )

        # Mix with original image based on opacity
        output = (1 - opacity) * src1 + opacity * blend

        # Convert back to 8-bit image
        output = (output * 255).astype(np.uint8)
        return output
    #调整色温
    def adjust_image(self, image):
        # 将图像转换为 HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 获取当前的色相、饱和度和亮度
        h, s, v = cv2.split(hsv_image)

        # 调整色相
        mean_hue = np.mean(h)
        if mean_hue > 70:
            target_hue = random.randint(30, 60)  # 20° 到 40° 映射到 0-179
            hue_adjustment = target_hue - mean_hue
            h = np.clip(h + hue_adjustment, 0, 179)

        # 调整饱和度
        mean_saturation = np.mean(s)
        if mean_saturation > 200:
            target_saturation = random.randint(76, 135)  # 30% 到 60%
            saturation_adjustment = target_saturation/2 - mean_saturation
            s = np.clip(s + saturation_adjustment, 0, 255)

        # # 调整亮度
        mean_value = np.mean(v)# 转换为 0-1 范围
        if mean_value > 200:
            target_value = random.randint(127, 179)  # 50% 到 70%
            value_adjustment = target_value - mean_value
            v = np.clip(v + value_adjustment, 0, 255)

        # 合并调整后的通道
        adjusted_hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])

        # 转换回 BGR
        adjusted_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
        return adjusted_image
    # 增加色温
    def increase_color_temperature(self, image, intensity=30):
        # Convert image to float32 for precision
        hue, saturation, bright = self.get_hue_saturation_and_value(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # img = image.astype(np.float32)
        if hue < 25 or hue >45:
            hue_adjust = hue.astype(np.uint8) - random.randint(25, 45)
            hsv_image[:, :, 0] = hsv_image[:, :, 0] - hue_adjust
        if saturation < 20 or saturation > 40:
            saturation_adjust = saturation.astype(np.uint8) - random.randint(25, 45)
            hsv_image[:, :, 1] = hsv_image[:, :, 1] - saturation_adjust
        if bright < 80 or bright > 120:
            bright_adjust = bright.astype(np.uint8) - random.randint(25, 45)
            hsv_image[:, :, 2] = hsv_image[:, :, 2] - bright_adjust
        hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # Create a warming filter by increasing red and green channels
        # warming_filter = np.array([1.0, 1.0, 1.0])
        # warming_filter[2] += intensity / 100.0  # Increase red
        # warming_filter[1] += intensity / 200.0  # Slightly increase green

        # Apply the warming filter
        # warmed_img = img * warming_filter

        # Clip to valid range [0, 255] and convert back to uint8
        # warmed_img = np.clip(warmed_img, 0, 255).astype(np.uint8)

        return output
    # 正片叠底模式
    def multiply_blend(self, src1, src2, opacity=0.5):
        if src1.shape != src2.shape:
            raise ValueError("Input images must have the same dimensions and type")

        # Convert images to float32 for precision
        src1 = src1.astype(np.float32) / 255.0
        src2 = src2.astype(np.float32) / 255.0
        # 进行正片叠底操作
        blend = cv2.multiply(src1, src2)
        # Mix with original image based on opacity
        output = (1 - opacity) * src1 + opacity * blend
        # Convert back to 8-bit image
        output = (output * 255).astype(np.uint8)
        return output
    # 将图片转为黑白
    def convert_to_black_and_white(self, src1):
        one_channel = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
        output = cv2.merge([one_channel, one_channel, one_channel])
        return output
    # 调整图片色相、饱和度、亮度
    def adjust_hue_saturation_bright(self, image, hue_shift, saturation_shift=None, brightness_shift=None):
        # 将BGR图像转换为HSV色彩空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 调整色相
        if saturation_shift is not None:
            hsv_image[:, :, 1] = saturation_shift
        hsv_image[:, :, 0] = hue_shift  # (hsv_image[:, :, 0] + hue_shift) % 180  # 色相在0-179之间
        if brightness_shift is not None:
            hsv_image[:, :, 2] = hsv_image[:, :, 2] + brightness_shift
            np.clip(hsv_image[:, :, 2], 0, 255).astype(np.uint8)
        # 将图像转换回BGR
        output = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return output
    # 获得图片的色相、饱和度和亮度
    def get_hue_saturation_and_value(self, image):
        # 将BGR图像转换为HSV色彩空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 提取色相、饱和度和亮度（值）通道
        hue = hsv_image[:, :, 0]
        saturation = hsv_image[:, :, 1]
        value = hsv_image[:, :, 2]

        # 计算平均色相、饱和度和亮度
        average_hue = np.mean(hue)
        average_saturation = np.mean(saturation)
        average_value = np.mean(value)

        return average_hue, average_saturation, average_value

    #随机裁减
    def random_crop(self, spot, GT_shape):
        # 获取目标图像的高度和宽度
        target_height, target_width = GT_shape[0], GT_shape[1]

        # 获取原始图像的高度和宽度
        height, width = spot.shape[:2]

        # 确保裁剪区域在原图像内
        max_x = width - target_width
        max_y = height - target_height

        if max_x < 0 or max_y < 0:
            return False, spot

        # 随机选择裁剪的起始点
        start_x = np.random.randint(0, max_x + 1)
        start_y = np.random.randint(0, max_y + 1)

        # 裁剪图像
        cropped_spot = spot[start_y:start_y + target_height, start_x:start_x + target_width]

        return True, cropped_spot

    def choose_random_file(self, directory):
        files = os.listdir(directory)
        file = random.choice(files)
        path = os.path.join(directory, file)
        return cv2.imread(path) if path else None, file[:-4]

    def __call__(self, GT):  #重写__call__，使得可以通过实例()调用该方法
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        #0、模糊
        # if random.random() < 0.25 and self.phase == 'train':
        #     noise_syn = GT
        # else:
        noise_syn = online_add_degradation_v2(GT)
        ###############################

        hot = np.abs(np.array(noise_syn).astype(np.float32) - np.array(GT).astype(np.float32))
        mask = np.max(hot, axis=2)
        mask = np.expand_dims((mask/255).astype(np.float32), axis=2)
        return noise_syn, mask, None
        ###############################
        GT = np.array(GT)[:, :, ::-1]  # 将进行将PIL转为numpy需要调换RGB顺序，PIL是RGB，OpenCV是BGR
        noise_syn = np.array(noise_syn)[:, :, ::-1]
        real_hot = np.abs(GT.astype(np.float32) - noise_syn.astype(np.float32))#加燥部分的热力图
        ##############################
        name = []
        scratch, scratch_name = self.choose_random_file('datasets/scratch')
        flaw, flaw_name = self.choose_random_file('datasets/flaw')
        spot, spot_name = self.choose_random_file('datasets/spot')

        # Ensure images are the same size
        scratch = cv2.resize(scratch, (GT.shape[1], GT.shape[0]))
        flaw = cv2.resize(flaw, (GT.shape[1], GT.shape[0]))

        spot_crop_flag, spot = self.random_crop(spot, GT.shape)
        if not spot_crop_flag:
            spot = cv2.resize(spot, (GT.shape[1], GT.shape[0]))
        #####################
        angle = random.randint(-30, 30)
        scratch = imutils.rotate(scratch, angle)
        angle = random.randint(-30, 30)
        flaw = imutils.rotate(flaw, angle)
        #####################

        ######################################
        if self.phase == 'train':
            low_board = 0.0
            high_board = 0.9
        else:
            low_board = 0.3
            high_board = 0.9

        # 1、污渍
        spot_syn = noise_syn
        spot_num = 3
        if self.spot_flag:
            opacity = random.uniform(low_board, high_board)
            if random.random() < 1/spot_num:
                spot_syn = self.multiply_blend(spot_syn, spot, opacity=opacity)  # 正片叠底
                name.extend(['spot', spot_name, '-multiply_blend-'])
            elif random.random() < 1 / spot_num:
                spot_syn = self.soft_light_blend(spot_syn, spot, opacity=opacity)  # 柔光
                name.extend(['spot', spot_name, '-soft_light_blend-'])
            elif random.random() < 1 / spot_num:
                spot_syn = self.overlay_blend(spot_syn, spot, opacity=opacity) # 叠加
                name.extend(['spot', spot_name, '-overlay_blend-'])
        real_hot = np.maximum(real_hot, np.abs(GT.astype(np.float32) - spot_syn.astype(np.float32)))  # 加燥部分的热力图
        # 0、转为黑白
        # result = convert_to_black_and_white(result)
        # 2、划痕
        scratch_syn = spot_syn
        scratch_num = 4
        if self.scratch_flag:
            opacity = random.uniform(low_board, high_board)
            if random.random() < 1 / scratch_num:
                scratch_syn = self.screen_blend(scratch_syn, scratch, opacity=opacity)  # 滤色
                name.extend(['scratch', scratch_name, '-screen_blend-'])
                real_hot = np.maximum(real_hot, (opacity * scratch).astype(np.float32))  # 加燥部分的热力图
            elif random.random() < 1 / scratch_num:
                scratch_syn = self.soft_light_blend(scratch_syn, scratch, opacity=opacity) # 柔光
                name.extend(['scratch', scratch_name, '-soft_light_blend-'])
                real_hot = np.maximum(real_hot, (opacity * scratch).astype(np.float32))  # 加燥部分的热力图
            elif random.random() < 1 / scratch_num:
                scratch_syn = self.brighten(scratch_syn, scratch, opacity=opacity) # 变亮
                name.extend(['scratch', scratch_name, '-brighten-'])
                real_hot = np.maximum(real_hot, (opacity * scratch).astype(np.float32))  # 加燥部分的热力图
            elif random.random() < 1 / scratch_num:
                scratch_syn = self.lighten(scratch_syn, scratch, opacity=opacity) #浅色
                name.extend(['scratch', scratch_name, '-lighten-'])
                real_hot = np.maximum(real_hot, (opacity * scratch).astype(np.float32))  # 加燥部分的热力图
        # 3、裂纹
        flaw_syn = scratch_syn
        flaw_num = 4
        if self.flwa_flag:
            opacity = random.uniform(low_board, high_board)
            if random.random() < 1 / flaw_num:
                flaw_syn = self.screen_blend(flaw_syn, flaw, opacity=opacity)#叠加
                name.extend(['flaw', flaw_name, '-screen_blend-'])
                real_hot = np.maximum(real_hot, (opacity * flaw).astype(np.float32))
            elif random.random() < 1 / flaw_num:
                flaw_syn = self.screen_blend(flaw_syn, flaw, opacity=opacity)#滤色
                name.extend(['flaw', flaw_name, '-screen_blend-'])
                real_hot = np.maximum(real_hot, (opacity * flaw).astype(np.float32))
            elif random.random() < 1 / flaw_num:
                flaw_syn = self.brighten(flaw_syn, flaw, opacity=opacity)#变亮
                name.extend(['flaw', flaw_name, '-brighten-'])
                real_hot = np.maximum(real_hot, (opacity * flaw).astype(np.float32))
            elif random.random() < 1 / flaw_num:
                flaw_syn = self.lighten(flaw_syn, flaw, opacity=opacity)
                name.extend(['flaw', flaw_name, '-lighten-'])
                real_hot = np.maximum(real_hot, (opacity * flaw).astype(np.float32))

        # 4、增加色温
        # result = self.adjust_image(result)
        # result = self.increase_color_temperature(result, intensity=30)
        # 4、调整色相
        # result = self.adjust_hue_saturation_bright(result, hue_shift=16, saturation_shift=78, brightness_shift=20)
        # real_hot = np.mean((np.abs(flaw_syn.astype(np.float32) - GT.astype(np.float32)) + real_hot)/2, axis=2)
        real_hot = np.mean(np.maximum(np.abs(flaw_syn.astype(np.float32) - GT.astype(np.float32)), real_hot), axis=2)
        hot_image = real_hot / 255
        name.extend(['.jpg'])
        return Image.fromarray(flaw_syn[:, :, ::-1]), np.expand_dims(hot_image.astype(np.float32), axis=2), name

if __name__ == '__main__':

    GT = Image.open('data/test/2007_004052.jpg').convert('RGB')
    ps = Photoshop(spot_flag=True, scratch_flag=True, flwa_flag=True, phase='test')
    Syn, mask, name = ps(GT)
    mask = Image.fromarray((mask*256).astype(np.uint8).squeeze(), mode="L")
    Syn.save('data/output/2007_004052.jpg')
    mask.save('data/output/mask.png')
