
import tensorflow as tf


def distorted_bounding_box_crop(
    img_bytes,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(0.75, 1.33),
    area_range=(0.05, 1.0),
    max_attempts=100,
):
  """Generates cropped_image using one of the bboxes randomly distorted.
  See `tf.image.sample_distorted_bounding_box` for more documentation.
  """
  shape = tf.io.extract_jpeg_shape(img_bytes)
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True,
  )

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.io.decode_and_crop_jpeg(img_bytes, crop_window, channels=3)

  return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  return tf.reduce_sum(tf.cast(a==b, tf.int32))>=x


def decode_and_random_crop(img_bytes, img_size):
  """Make a random crop of img_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      img_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3.0 / 4, 4.0 / 3.0),
      area_range=(0.08, 1.0),
      max_attempts=10,
  )
  original_shape = tf.io.extract_jpeg_shape(img_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: decode_and_center_crop(img_bytes, img_size),
      lambda: tf.image.resize([image], [img_size, img_size],
                              method=tf.image.ResizeMethod.BICUBIC)[0],
  )

  return image


def decode_and_center_crop(img_bytes, img_size, crop_padding=32):
  """Crops to center of image with padding then scales img_size."""
  shape = tf.io.extract_jpeg_shape(img_bytes)
  img_h, img_w = shape[0], shape[1]

  pad_cntcrp_size = tf.cast(
      (
          (img_size / (img_size + crop_padding))
          * tf.cast(tf.minimum(img_h, img_w), tf.float32)
      ),
      tf.int32,
  )

  offset_h = ((img_h - pad_cntcrp_size) + 1) // 2
  offset_w = ((img_w - pad_cntcrp_size) + 1) // 2
  crop_window = tf.stack([
      offset_h,
      offset_w,
      pad_cntcrp_size,
      pad_cntcrp_size,
  ])
  image = tf.io.decode_and_crop_jpeg(img_bytes, crop_window, channels=3)
  image = tf.image.resize([image], [img_size, img_size],
                          method=tf.image.ResizeMethod.BICUBIC)[0]

  return image