import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import mnist
import argparse
import json
import sys

class MegapixelMNIST:
    """
    Class to create an artificial megapixel MNIST dataset
    """

    class Sample(object):
        def __init__(self, dataset, idxs, positions, noise_positions, noise_patterns):
            self._dataset = dataset
            self._idxs = idxs
            self._positions = positions
            self._noise_positions = noise_positions
            self._noise_patterns = noise_patterns
            self._img = None

        @property
        def noise_positions_and_patterns(self):
            return zip(self._noise_positions, self._noise_patterns)

        def _get_slice(self, pos, s, scale=1, offset=(0, 0)):
            pos = (int(pos[0] * scale - offset[0]), int(pos[1] * scale - offset[1]))
            s = int(s)
            return (
                slice(max(0, pos[0]), max(0, pos[0] + s)),
                slice(max(0, pos[1]), max(0, pos[1] + s)),
                0
            )

        def create_img(self):
            if self._img is None:
                size = self._dataset._H, self._dataset._W
                img = np.zeros(size + (1,), dtype=np.uint8)
                
                if self._dataset._should_add_noise:
                    for p, i in self.noise_positions_and_patterns:
                        img[self._get_slice(p, self._dataset._noise_size)] = 255 * self._dataset._noise[i]
                
                for p, i in zip(self._positions, self._idxs):
                    img[self._get_slice(p, self._dataset._digit_size)] = 255 * self._dataset._images[i]
                
                self._img = img
            return self._img

    def __init__(self, N=5000, W=1500, H=1500, train=True, noise=True, n_noise=50, seed=0, digit_size=28, noise_size=28):
        # Load the images
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = x_train if train else x_test
        y = y_train if train else y_test
        x = x.astype(np.float32) / 255.

        self._digit_size = digit_size
        self._noise_size = noise_size

        self._W, self._H = W, H
        self._images = self.resize_images(x, self._digit_size)

        # Generate the dataset
        try:
            random_state = np.random.get_state()
            np.random.seed(seed + int(train))
            self._nums, self._targets, self._digits, self._max_targets = self._get_numbers(N, y)
            self._pos = self._get_positions(N, W, H)
            self._top_targets = self._get_top_targets()
            self._noise, self._noise_positions, self._noise_patterns = self._create_noise(N, W, H, n_noise)
        finally:
            np.random.set_state(random_state)

        # Boolean whether to add noise
        self._should_add_noise = noise

    def _create_noise(self, N, W, H, n_noise):
        """
        Create some random scribble noise of curved lines and scale them
        """
        def generate_curved_line(image_size, num_points):
            """
            Generates a random curved line using Bezier curves
            """
            img = np.zeros((image_size, image_size), dtype=np.float32)
            thickness = int(2.121 * (28 / image_size))

            # Create random points for the Bezier curve
            points = np.random.rand(num_points, 2) * image_size
            points = points.astype(np.int32)

            # Define the Bezier curve function
            def bezier(t, points):
                n = len(points) - 1
                return sum(
                    (np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))) *
                    (1 - t)**(n - i) * t**i * points[i] for i in range(n + 1)
                )

            t = np.linspace(0, 1, 100)
            curve = bezier(t[:, np.newaxis], points)

            # Draw the curve on the image
            for i in range(len(curve) - 1):
                start_point = tuple(curve[i].astype(int))
                end_point = tuple(curve[i + 1].astype(int))
                cv2.line(img, start_point, end_point, 1, thickness)

            return img

        noise_patterns = []
        for _ in range(n_noise):
            num_points = np.random.choice([4, 6, 8], p=[3/9, 5/9, 1/9])
            noise_patterns.append(generate_curved_line(28, num_points))

        noise_patterns = np.array(noise_patterns)
        noise_patterns = self.resize_images(noise_patterns, self._noise_size)

        # Randomly assign noise to all images
        positions = (np.random.rand(N, n_noise, 2) * [H - self._noise_size, W - self._noise_size]).astype(int)
        patterns = (np.random.rand(N, n_noise) * n_noise).astype(int)

        return noise_patterns, positions, patterns

    def resize_images(self, images, new_size):
        resized_images = []
        for img in images:
            resized_img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)
            resized_images.append(resized_img)
        return np.array(resized_images)

    def _get_numbers(self, N, y):
        nums = []
        targets = []
        max_targets = []
        all_digits = []
        all_idxs = np.arange(len(y))
        for _ in range(N):
            target = int(np.random.rand() * 10)
            positive_idxs = np.random.choice(all_idxs[y == target], 3)
            neg_idxs = np.random.choice(all_idxs[y != target], 2)
            pos_neg_idxs = np.concatenate([positive_idxs, neg_idxs])
            digits = y[pos_neg_idxs]
            max_target = np.max(digits)

            nums.append(pos_neg_idxs)
            targets.append(target)
            all_digits.append(digits)
            max_targets.append(max_target)

        return np.array(nums), np.array(targets), np.array(all_digits), np.array(max_targets)

    def _get_positions(self, N, W, H):
        def overlap(positions, pos):
            if len(positions) == 0:
                return False
            distances = np.abs(np.asarray(positions) - np.asarray(pos)[np.newaxis])
            axis_overlap = distances < self._digit_size
            return np.logical_and(axis_overlap[:, 0], axis_overlap[:, 1]).any()

        positions = []
        for _ in range(N):
            position = []
            for _ in range(5):
                while True:
                    pos = np.round(np.random.rand(2) * [H - self._digit_size, W - self._digit_size]).astype(int)
                    if not overlap(position, pos):
                        break
                position.append(pos)
            positions.append(position)

        return np.array(positions)
    
    def _get_top_targets(self):
        pos_height = self._pos[:, :, 0] 
        top_pos_idx = np.argmin(pos_height, axis=-1)
        N = self._digits.shape[0]
        top_targets = self._digits[np.arange(N), top_pos_idx]
        return top_targets

    def __len__(self):
        return len(self._nums)

    def __getitem__(self, i):
        if len(self) <= i:
            raise IndexError()
        sample = self.Sample(
            self,
            self._nums[i],
            self._pos[i],
            self._noise_positions[i],
            self._noise_patterns[i]
        )
        x = sample.create_img().astype(np.float32) / 255
        y = self._targets[i]
        y_max = self._max_targets[i]
        y_top = self._top_targets[i]
        y_multi = np.eye(10)[self._digits[i]].sum(0).clip(0, 1)
        return x, y, y_max, y_top, y_multi

def sparsify(dataset):
    def to_sparse(x):
        x = x.ravel()
        indices = np.where(x != 0)[0]
        values = x[indices]
        return (indices, values)

    print("Sparsifying dataset")
    data = []
    for i, (x, y_maj, y_max, y_top, y_multi) in enumerate(dataset):
        print(
            "\u001b[1000DProcessing {:5d} / {:5d}".format(i + 1, len(dataset)),
            end="",
            flush=True
        )

        data.append({
            'input': to_sparse(x),
            'majority': y_maj,
            'max': y_max,
            'top': y_top,
            'multi': y_multi
        })
    print()
    return data

def main(argv):
    parser = argparse.ArgumentParser(
        description="Create the Megapixel MNIST dataset"
    )
    parser.add_argument(
        "digit_size",
        type=int,
        help="Pixel size of the digit within the canvas"
    )
    parser.add_argument(
        "noise_size",
        type=int,
        help="Pixel size of the noise pattern within the canvas"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=5000,
        help="How many images to create for training set"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=1000,
        help="How many images to create for test set"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1500,
        help="Set the width for the image"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1500,
        help="Set the height for the image"
    )
    parser.add_argument(
        "--no_noise",
        action="store_false",
        dest="noise",
        help="Do not use noise in the dataset"
    )
    parser.add_argument(
        "--n_noise",
        type=int,
        default=50,
        help="Set the number of noise patterns per image"
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=0,
        help="Choose the random seed for the dataset"
    )
    parser.add_argument(
        "output_directory",
        help="The directory to save the dataset into"
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    with open(os.path.join(args.output_directory, "parameters.json"), "w") as f:
        json.dump(
            {
                "n_train": args.n_train,
                "n_test": args.n_test,
                "width": args.width,
                "height": args.height,
                "noise": args.noise,
                "n_noise": args.n_noise,
                "seed": args.dataset_seed,
                "noise_size": args.noise_size,
                "digit_size": args.digit_size
            },
            f,
            indent=4
        )

    # Write the training set
    training = MegapixelMNIST(
        N=args.n_train,
        train=True,
        W=args.width,
        H=args.height,
        noise=args.noise,
        n_noise=args.n_noise,
        seed=args.dataset_seed,
        digit_size=args.digit_size,
        noise_size=args.noise_size
    )
    data = sparsify(training)
    np.save(os.path.join(args.output_directory, "train.npy"), data)

    # Write the test set
    test = MegapixelMNIST(
        N=args.n_test,
        train=False,
        W=args.width,
        H=args.height,
        noise=args.noise,
        n_noise=args.n_noise,
        seed=args.dataset_seed,
        digit_size=args.digit_size,
        noise_size=args.noise_size
    )
    data = sparsify(test)
    np.save(os.path.join(args.output_directory, "test.npy"), data)

# Usage example: python make_mnist.py --width 1500 --height 1500 dsets/megapixel_mnist_1500
if __name__ == "__main__":
    main(sys.argv[1:])

