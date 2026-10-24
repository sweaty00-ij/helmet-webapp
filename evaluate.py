"""ประเมินโมเดลบน Test set แล้วบันทึกผลไว้ใน results/ (หน้า Evaluation ของเว็บอ่านจากที่นี่)

ใช้:  python evaluate.py --test path/to/test
      โฟลเดอร์ test ต้องมี images/ และ labels/ แบบ YOLO (ดาวน์โหลดจาก Roboflow ฟอร์แมต yolov11)
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
PLOTS = ["confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="โฟลเดอร์ test ที่มี images/ และ labels/")
    parser.add_argument("--model", default=str(ROOT / "yolo_10epoch_best.pt"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    test_images = Path(args.test).resolve() / "images"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = [model.names[i] for i in range(len(model.names))]

    with tempfile.TemporaryDirectory() as tmp:

        # data.yaml ชั่วคราว ชี้ทุก split ไปที่ test (ใช้แค่ split="test")
        data_yaml = Path(tmp) / "data.yaml"
        data_yaml.write_text(yaml.safe_dump({
            "train": str(test_images), "val": str(test_images), "test": str(test_images),
            "nc": len(names), "names": names,
        }))

        res = model.val(data=str(data_yaml), split="test", imgsz=640, project=tmp, name="val", plots=True)

        for plot in PLOTS:
            src = Path(res.save_dir) / plot
            if src.exists():
                shutil.copy(src, out / plot)

    per_class = {}
    for i, c in enumerate(res.box.ap_class_index):
        p, r, ap50, ap = res.box.class_result(i)
        per_class[names[int(c)]] = {"precision": p, "recall": r, "AP50": ap50, "AP50-95": ap}

    metrics = {
        "model": Path(args.model).name,
        "test_images": sum(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} for p in test_images.iterdir()),
        "mAP50-95": res.box.map,
        "mAP50": res.box.map50,
        "mAP75": res.box.map75,
        "precision": res.box.mp,
        "recall": res.box.mr,
        "per_class": per_class,
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
