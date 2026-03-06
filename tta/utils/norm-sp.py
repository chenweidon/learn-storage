import os
import cv2
import numpy as np
from PIL import Image
import argparse
import shutil
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ImageNormalizer:
    def __init__(self, target_size=None, normalize_pixels=True, keep_aspect_ratio=False):
        """
        图像归一化处理器

        参数:
        target_size: 目标尺寸 (width, height)，如果为None则保持原尺寸
        normalize_pixels: 是否对像素值进行归一化（0-1范围）
        keep_aspect_ratio: 是否保持宽高比（仅在调整大小时有效）
        """
        self.target_size = target_size
        self.normalize_pixels = normalize_pixels
        self.keep_aspect_ratio = keep_aspect_ratio
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

    def normalize_image(self, image_path, output_path, format='jpg'):
        """归一化单个图像"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                # 尝试用PIL读取
                img_pil = Image.open(image_path)
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            if img is None:
                print(f"无法读取图像: {image_path}")
                return False

            original_size = img.shape[:2]  # (height, width)
            original_dtype = img.dtype

            # 1. 调整尺寸
            if self.target_size is not None:
                img = self._resize_image(img)

            # 2. 像素值归一化
            if self.normalize_pixels:
                img = self._normalize_pixel_values(img)

            # 3. 保存图像
            success = self._save_image(img, output_path, format, original_dtype)

            if success:
                return True, original_size, img.shape[:2]
            else:
                return False, None, None

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return False, None, None

    def _resize_image(self, img):
        """调整图像尺寸"""
        h, w = img.shape[:2]
        target_w, target_h = self.target_size

        if self.keep_aspect_ratio:
            # 保持宽高比，填充到目标尺寸
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 调整大小
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 创建目标图像并填充
            if len(img.shape) == 3:  # 彩色图像
                target_img = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
            else:  # 灰度图像
                target_img = np.zeros((target_h, target_w), dtype=img.dtype)

            # 计算填充位置（居中）
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2

            target_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            return target_img
        else:
            # 直接调整到目标尺寸
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _normalize_pixel_values(self, img):
        """归一化像素值到0-1范围"""
        if img.dtype == np.uint8:
            img_normalized = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img_normalized = img.astype(np.float32) / 65535.0
        else:
            img_normalized = img.astype(np.float32)

        # 确保在0-1范围内
        img_normalized = np.clip(img_normalized, 0, 1)
        return img_normalized

    def _save_image(self, img, output_path, format, original_dtype):
        """保存图像"""
        try:
            # 如果归一化了像素值，转换回0-255范围
            if self.normalize_pixels and img.dtype != np.uint8:
                img_to_save = (img * 255).astype(np.uint8)
            else:
                img_to_save = img.astype(np.uint8)

            # 保存图像
            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(output_path, img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif format.lower() == 'png':
                cv2.imwrite(output_path, img_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(output_path, img_to_save)

            return True
        except Exception as e:
            print(f"保存图像 {output_path} 时出错: {e}")
            return False

    def process_folder(self, input_folder, output_folder, start_index=1,
                       output_format='jpg', copy_unsupported=True):
        """
        处理整个文件夹中的图像

        参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        start_index: 起始编号
        output_format: 输出格式
        copy_unsupported: 是否复制不支持的文件
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 获取所有文件
        all_files = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

        # 按名称排序
        all_files.sort()

        print(f"找到 {len(all_files)} 个文件")
        print(f"开始处理，从 {start_index:05d} 开始编号")
        print(f"输出格式: {output_format}")
        print("=" * 80)

        processed_count = 0
        skipped_count = 0
        supported_count = 0
        unsupported_count = 0

        # 统计信息
        original_sizes = []
        new_sizes = []

        for i, file_path in enumerate(all_files):
            # 检查文件扩展名
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext in self.supported_formats:
                supported_count += 1
                # 生成输出文件名
                new_name = f"{start_index + processed_count:05d}.{output_format}"
                output_path = os.path.join(output_folder, new_name)

                # 处理图像
                success, original_size, new_size = self.normalize_image(
                    file_path, output_path, output_format
                )

                if success:
                    processed_count += 1
                    original_sizes.append(original_size)
                    new_sizes.append(new_size)

                    # 打印进度
                    if processed_count % 10 == 0 or i == len(all_files) - 1:
                        print(f"已处理: {processed_count}/{supported_count} ({i + 1}/{len(all_files)})")

                        # 显示示例文件名
                        if processed_count <= 5 or i == len(all_files) - 1:
                            original_name = os.path.basename(file_path)
                            print(f"  {original_name} -> {new_name}")
                else:
                    skipped_count += 1
                    print(f"跳过: {os.path.basename(file_path)}")
            else:
                unsupported_count += 1
                if copy_unsupported:
                    # 复制不支持的文件到输出文件夹
                    new_name = f"{start_index + processed_count:05d}{file_ext}"
                    output_path = os.path.join(output_folder, new_name)
                    shutil.copy2(file_path, output_path)
                    processed_count += 1
                    print(f"复制不支持的文件: {os.path.basename(file_path)} -> {new_name}")

        # 打印统计信息
        print("=" * 80)
        print(f"处理完成!")
        print(f"总文件数: {len(all_files)}")
        print(f"支持的图像文件: {supported_count}")
        print(f"不支持的格式: {unsupported_count}")
        print(f"成功处理: {processed_count}")
        print(f"跳过: {skipped_count}")

        if original_sizes:
            # 计算尺寸统计
            original_heights = [s[0] for s in original_sizes]
            original_widths = [s[1] for s in original_sizes]

            new_heights = [s[0] for s in new_sizes]
            new_widths = [s[1] for s in new_sizes]

            print("\n尺寸统计:")
            print(f"原始图像 - 平均尺寸: {np.mean(original_widths):.0f}x{np.mean(original_heights):.0f}")
            print(f"原始图像 - 最小尺寸: {np.min(original_widths)}x{np.min(original_heights)}")
            print(f"原始图像 - 最大尺寸: {np.max(original_widths)}x{np.max(original_heights)}")

            if self.target_size:
                print(f"目标尺寸: {self.target_size[0]}x{self.target_size[1]}")
                print(f"处理后的图像 - 平均尺寸: {np.mean(new_widths):.0f}x{np.mean(new_heights):.0f}")

        return processed_count


def main():
    parser = argparse.ArgumentParser(description='图像归一化和重命名工具')
    parser.add_argument('--input', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--size', type=str, default=None, help='目标尺寸，格式: "宽度,高度"，如 "224,224"')
    parser.add_argument('--start', type=int, default=1, help='起始编号，默认: 1')
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'png', 'bmp', 'tiff'],
                        help='输出格式，默认: jpg')
    parser.add_argument('--no-normalize', action='store_true', help='不进行像素值归一化')
    parser.add_argument('--keep-aspect', action='store_true', help='调整大小时保持宽高比')
    parser.add_argument('--no-copy-unsupported', action='store_true', help='不复制不支持格式的文件')

    args = parser.parse_args()

    # 解析尺寸参数
    target_size = None
    if args.size:
        try:
            w, h = map(int, args.size.split(','))
            target_size = (w, h)
        except ValueError:
            print("错误: 尺寸格式应为 '宽度,高度'，如 '224,224'")
            return

    # 创建归一化器
    normalizer = ImageNormalizer(
        target_size=target_size,
        normalize_pixels=not args.no_normalize,
        keep_aspect_ratio=args.keep_aspect
    )

    # 处理文件夹
    processed = normalizer.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        start_index=args.start,
        output_format=args.format,
        copy_unsupported=not args.no_copy_unsupported
    )

    print(f"\n所有文件已保存到: {args.output}")


# 简化版本 - 直接在代码中配置使用
def normalize_images_simple(input_folder, output_folder, start_index=1, target_size=None):
    """
    简化版本的图像归一化函数

    参数:
    input_folder: 输入文件夹路径
    output_folder: 输出文件夹路径
    start_index: 起始编号
    target_size: 目标尺寸 (宽度, 高度)
    """

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']

    # 获取所有图像文件
    image_files = []
    for file in sorted(os.listdir(input_folder)):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_folder, file))

    print(f"找到 {len(image_files)} 张图片")
    print(f"开始归一化处理...")

    for i, img_path in enumerate(image_files):
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取: {img_path}")
            continue

        # 调整尺寸
        if target_size:
            h, w = target_size[1], target_size[0]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # 归一化像素值到0-1
        img_normalized = img.astype(np.float32) / 255.0

        # 转换回0-255范围保存
        img_to_save = (img_normalized * 255).astype(np.uint8)

        # 生成新文件名
        new_name = f"{start_index + i:05d}.jpg"
        output_path = os.path.join(output_folder, new_name)

        # 保存图像
        cv2.imwrite(output_path, img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 打印进度
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"已处理: {i + 1}/{len(image_files)} - {os.path.basename(img_path)} -> {new_name}")

    print(f"\n处理完成! 所有图像已保存到: {output_folder}")
    print(f"文件命名从 {start_index:05d}.jpg 到 {start_index + len(image_files) - 1:05d}.jpg")


# 高级功能：保持目录结构
def normalize_images_with_structure(input_folder, output_folder, start_index=1):
    """
    保持原始目录结构进行归一化和重命名
    """

    def process_subfolder(subfolder_path, output_base, current_index):
        """处理子文件夹"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []

        for file in sorted(os.listdir(subfolder_path)):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(subfolder_path, file))

        # 创建对应的输出子文件夹
        relative_path = os.path.relpath(subfolder_path, input_folder)
        output_subfolder = os.path.join(output_base, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for i, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 生成新文件名
            new_name = f"{current_index:05d}.jpg"
            output_path = os.path.join(output_subfolder, new_name)

            # 保存图像（这里可以添加归一化处理）
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            current_index += 1

        return current_index

    # 遍历所有子文件夹
    current_index = start_index
    for root, dirs, files in os.walk(input_folder):
        current_index = process_subfolder(root, output_folder, current_index)

    print(f"处理完成! 总共处理了 {current_index - start_index} 张图片")


if __name__ == "__main__":
    # 方法1: 使用命令行参数
    # python image_normalizer.py --input /path/to/input --output /path/to/output --size 224,224 --start 1

    # 方法2: 直接在代码中配置
    input_folder = "F:\\cwd\\tta_mmf\\data\\comparison-sp\\input"  # 替换为你的输入文件夹路径
    output_folder = "F:\\cwd\\tta_mmf\\data\\comparison-sp\\input-norm"  # 替换为你的输出文件夹路径

    if os.path.exists(input_folder):
        # 使用简化版本
        # normalize_images_simple(
        #     input_folder=input_folder,
        #     output_folder=output_folder,
        #     start_index=1,
        #     target_size=(384, 384)  # 可选：调整到指定尺寸
        # )

        # 或者使用完整版本
        normalizer = ImageNormalizer(target_size=(384, 384))
        normalizer.process_folder(input_folder, output_folder, start_index=1)
    else:
        print("请通过命令行参数指定文件夹路径:")
        print("python image_normalizer.py --input /path/to/input --output /path/to/output")
        print("\n或者修改脚本中的 input_folder 和 output_folder 变量")

        # 运行命令行版本
        main()