import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOv10

if __name__ == '__main__':
    model = YOLOv10(model=r'ultralytics\cfg\models\v10\yolov10n.yaml')
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=5,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='train_v10',
                single_cls=False,
                cache=False,
                )
