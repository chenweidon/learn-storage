import os
import cv2
import numpy as np
import argparse
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class ImageComparator:
    def __init__(self):
        self.results = []

    def load_image(self, img_path, convert_to_grayscale=True):
        """加载图像，可选择转换为灰度图"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                # 如果OpenCV读取失败，尝试用PIL
                img_pil = Image.open(img_path)
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            if convert_to_grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif not convert_to_grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            return None

    def calculate_pcc(self, img1, img2):
        """计算皮尔逊相关系数"""
        # 确保图像有相同尺寸
        if img1.shape != img2.shape:
            # 调整图像尺寸为相同
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = img1[:h, :w]
            img2 = img2[:h, :w]

        # 展平图像
        flat1 = img1.flatten().astype(float)
        flat2 = img2.flatten().astype(float)

        # 计算PCC
        pcc, _ = stats.pearsonr(flat1, flat2)
        return pcc

    def calculate_ssim(self, img1, img2):
        """计算结构相似性指数"""
        # 确保图像有相同尺寸
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = img1[:h, :w]
            img2 = img2[:h, :w]

        # 计算SSIM
        # 对于彩色图像，使用多通道SSIM
        if len(img1.shape) == 3:
            ssim_val = ssim(img1, img2, channel_axis=2, data_range=255)
        else:
            ssim_val = ssim(img1, img2, data_range=255)
        return ssim_val

    def compare_folders(self, folder1, folder2, num_images=None, grayscale=True, start_from=0):
        """比较两个文件夹中的图像"""
        # 获取图像文件列表
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']

        # 获取文件夹1中的图像文件
        images1 = []
        for root, dirs, files in os.walk(folder1):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    images1.append(os.path.join(root, file))
        images1.sort()

        # 获取文件夹2中的图像文件
        images2 = []
        for root, dirs, files in os.walk(folder2):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    images2.append(os.path.join(root, file))
        images2.sort()

        # 确定要比较的图像数量
        if num_images is None:
            num_images = min(len(images1), len(images2))
        else:
            num_images = min(num_images, len(images1), len(images2))

        print(f"文件夹1: {folder1} - 找到 {len(images1)} 张图片")
        print(f"文件夹2: {folder2} - 找到 {len(images2)} 张图片")
        print(f"开始从第 {start_from + 1} 张图片比较，共比较 {num_images} 张图片")
        print("=" * 80)
        print(f"{'序号':<6} {'文件名1':<30} {'文件名2':<30} {'PCC':<12} {'SSIM':<12}")
        print("-" * 80)

        # 比较图像
        pcc_values = []
        ssim_values = []

        for i in range(start_from, start_from + num_images):
            if i >= len(images1) or i >= len(images2):
                break

            img1_path = images1[i]
            img2_path = images2[i]

            img1_name = os.path.basename(img1_path)
            img2_name = os.path.basename(img2_path)

            # 加载图像
            img1 = self.load_image(img1_path, grayscale)
            img2 = self.load_image(img2_path, grayscale)

            if img1 is None or img2 is None:
                print(f"{i + 1:<6} {img1_name:<30} {img2_name:<30} {'加载失败':<12} {'加载失败':<12}")
                continue

            # 计算指标
            pcc = self.calculate_pcc(img1, img2)
            ssim_val = self.calculate_ssim(img1, img2)

            # 记录结果
            pcc_values.append(pcc)
            ssim_values.append(ssim_val)
            self.results.append({
                'index': i + 1,
                'file1': img1_name,
                'file2': img2_name,
                'pcc': pcc,
                'ssim': ssim_val
            })

            # 打印结果
            print(f"{i + 1:<6} {img1_name:<30} {img2_name:<30} {pcc:<12.4f} {ssim_val:<12.4f}")

        # 打印统计信息
        if pcc_values and ssim_values:
            print("=" * 80)
            print(f"统计信息:")
            print(f"平均PCC: {np.mean(pcc_values):.4f}")
            print(f"平均SSIM: {np.mean(ssim_values):.4f}")
            print(f"PCC标准差: {np.std(pcc_values):.4f}")
            print(f"SSIM标准差: {np.std(ssim_values):.4f}")
            print(f"最大PCC: {np.max(pcc_values):.4f}")
            print(f"最大SSIM: {np.max(ssim_values):.4f}")
            print(f"最小PCC: {np.min(pcc_values):.4f}")
            print(f"最小SSIM: {np.min(ssim_values):.4f}")

            # 按PCC排序显示最佳和最差对比
            print("\nPCC最高的5张图片:")
            sorted_by_pcc = sorted(self.results, key=lambda x: x['pcc'], reverse=True)
            for item in sorted_by_pcc[:5]:
                print(f"  {item['index']:<4} {item['file1']:<25} PCC: {item['pcc']:.4f}")

            print("\nSSIM最高的5张图片:")
            sorted_by_ssim = sorted(self.results, key=lambda x: x['ssim'], reverse=True)
            for item in sorted_by_ssim[:5]:
                print(f"  {item['index']:<4} {item['file1']:<25} SSIM: {item['ssim']:.4f}")

        return self.results

    def save_results_to_csv(self, output_file="comparison_results.csv"):
        """保存结果到CSV文件"""
        import csv

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '文件名1', '文件名2', 'PCC', 'SSIM'])
            for result in self.results:
                writer.writerow([
                    result['index'],
                    result['file1'],
                    result['file2'],
                    f"{result['pcc']:.6f}",
                    f"{result['ssim']:.6f}"
                ])
        print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='比较两个文件夹中图像的PCC和SSIM')
    parser.add_argument('--folder1', type=str, required=True, help='第一个文件夹路径')
    parser.add_argument('--folder2', type=str, required=True, help='第二个文件夹路径')
    parser.add_argument('--num', type=int, default=None, help='要比较的图像数量(默认:全部)')
    parser.add_argument('--start', type=int, default=0, help='从第几张开始比较(默认:0)')
    parser.add_argument('--color', action='store_true', help='使用彩色图像(默认:灰度)')
    parser.add_argument('--save', action='store_true', help='保存结果到CSV文件')
    parser.add_argument('--output', type=str, default="comparison_results.csv",
                        help='输出CSV文件名(默认:comparison_results.csv)')

    args = parser.parse_args()

    # 检查文件夹是否存在
    if not os.path.exists(args.folder1):
        print(f"错误: 文件夹 '{args.folder1}' 不存在")
        return

    if not os.path.exists(args.folder2):
        print(f"错误: 文件夹 '{args.folder2}' 不存在")
        return

    # 创建比较器并执行比较
    comparator = ImageComparator()
    results = comparator.compare_folders(
        folder1=args.folder1,
        folder2=args.folder2,
        num_images=args.num,
        grayscale=not args.color,
        start_from=args.start
    )

    # 如果需要保存结果
    if args.save and results:
        comparator.save_results_to_csv(args.output)


if __name__ == "__main__":
    # 如果没有使用命令行参数，可以在这里直接设置
    # 或者直接运行下面的示例代码

    # 示例使用代码
    folder1 = "F:\\cwd\\tta_mmf\\data\\comparison-sp\\input-norm"  # 替换为你的第一个文件夹路径
    folder2 = "F:\\cwd\\ctm_test\\data\\gama=36\\input-norm"  # 替换为你的第二个文件夹路径

    # 检查示例路径是否存在，如果不存在则提示
    if os.path.exists(folder1) and os.path.exists(folder2):
        comparator = ImageComparator()
        # 比较前10张图片
        results = comparator.compare_folders(folder1, folder2, num_images=10)
        # 保存结果
        comparator.save_results_to_csv()
    else:
        print("请通过命令行参数指定文件夹路径:")
        print("python image_comparator.py --folder1 /path/to/folder1 --folder2 /path/to/folder2 --num 20")
        print("\n或者修改脚本中的folder1和folder2变量")

        # 或者使用命令行参数解析
        main()