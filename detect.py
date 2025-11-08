import sys
import os
from pathlib import Path
from ultralytics import YOLOv10

# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt


model = YOLOv10(r"last.pt")


def gather_image_paths(source):
	"""If source is a directory, recursively collect image file paths.
	Otherwise return the original source (file path or list).
	"""
	exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
	# if list/tuple provided, assume it's already good
	if isinstance(source, (list, tuple)):
		return list(source)

	p = Path(source)
	if p.is_dir():
		imgs = [str(fp) for fp in p.rglob('*') if fp.suffix.lower() in exts]
		imgs.sort()
		return imgs

	# single file or other supported input
	return source


def main():
	# allow passing source via command line: python detect.py <source_path>
	default_source = r''
	src = sys.argv[1] if len(sys.argv) > 1 else default_source

	imgs = gather_image_paths(src)
	if isinstance(imgs, list) and len(imgs) == 0:
		print(f"No images found in directory: {src}")
		return

	# model.predict accepts a single path, a list of paths, or other Ultralytics-supported sources
	# To avoid allocating many large images at once (which can raise ArrayMemoryError),
	# predict one image at a time when imgs is a list.
	if isinstance(imgs, list):
		total = len(imgs)
		for idx, img_path in enumerate(imgs, start=1):
			try:
				print(f"Predicting ({idx}/{total}): {img_path}")
				model.predict(source=img_path, save=True)
			except MemoryError as me:
				print(f"MemoryError while predicting {img_path}: {me}")
				print("Try resizing large images or running on a machine with more memory.")
			except Exception as e:
				# catch numpy ArrayMemoryError and other unexpected errors
				if 'Memory' in str(e) or 'ArrayMemoryError' in str(type(e)):
					print(f"Out-of-memory error for {img_path}: {e}")
					print("Consider downsampling this image before predicting.")
				else:
					print(f"Error predicting {img_path}: {e}")
	else:
		model.predict(source=imgs, save=True)


if __name__ == '__main__':
	main()
