import matplotlib.pyplot as plt
from lab1.processor import ImgProcessor, CorrectionType

if __name__ == '__main__':
    # Task 1
    img = ImgProcessor('data/fog.png')
    plt.imshow(img.rgb_to_display(img.img))
    plt.show()

    # Task 2
    hsv_img = img.correction(CorrectionType.LINEAR)
    plt.imshow(img.hsv_to_display(hsv_img))
    plt.show()

    hsv_img = img.correction(CorrectionType.EXPONENTIAL)
    plt.imshow(img.hsv_to_display(hsv_img))
    plt.show()

    # Task 3
    filtered_image = img.apply_filter(ImgProcessor.gaussian_kernel, size=3,
                                      mean=0.5, std=1.0)
    plt.imshow(img.rgb_to_display(filtered_image))
    plt.show()

    filtered_image = img.apply_filter(ImgProcessor.box_kernel, size=3)
    plt.imshow(img.rgb_to_display(filtered_image))
    plt.show()

    filtered_image = img.apply_filter(ImgProcessor.unsharp_masking_kernel)
    plt.imshow(img.rgb_to_display(filtered_image))
    plt.show()

    # Task 4
    edges_x, edges_y = img.sobel_edges()
    plt.imshow(img.rgb_to_display(edges_x))
    plt.show()
    plt.imshow(img.rgb_to_display(edges_y))
    plt.show()
