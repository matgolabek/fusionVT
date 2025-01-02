import cv2
import numpy as np


def join_images_grid_with_index(image_paths, rows, cols):
    # Load all images
    images = [cv2.imread(path) for path in image_paths]

    # Resize all images to the same size
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]

    # Add index to each image
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)
    for idx, img in enumerate(resized_images):
        cv2.putText(
            img, f"{idx + 1}", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA
        )

    # Create the grid
    grid = []
    for i in range(rows):
        row_images = resized_images[i * cols:(i + 1) * cols]
        grid.append(np.hstack(row_images))  # Combine images in a row
    grid_image = np.vstack(grid)  # Combine rows into the final grid

    return grid_image


# Example usage
image_paths = ["track/%d.png" % i for i in range(190, 199)]

grid_image = join_images_grid_with_index(image_paths, rows=4, cols=2)
cv2.imwrite("human_track.jpg", grid_image)
cv2.imshow("Grid 2x4", grid_image)
cv2.waitKey(0)
