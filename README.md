# Smart Helmet Detection

Web application ตรวจว่าผู้ขับขี่มอเตอร์ไซค์สวมหมวกกันน็อกหรือไม่ จากภาพหรือวิดีโอ
ใช้ YOLO11 + OpenCV + Streamlit (วิชา Image Processing and Computer Vision)

**Demo:** https://final-img-helmet.streamlit.app

โค้ด train โมเดล (Week 1) และ Python application (Week 2) อยู่ใน repo ของกลุ่ม

## How to Run

ต้องใช้ Python 3.10–3.13

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

เปิดเบราว์เซอร์ที่ http://localhost:8501

## การใช้งาน

1. แถบด้านซ้าย เลือก Input เป็น **Image** หรือ **Video**
2. อัปโหลดไฟล์
   - ภาพ: `.jpg .jpeg .png .bmp .webp` ได้หลายไฟล์
   - วิดีโอ: `.mp4 .avi .mov .mkv .m4v .wmv` จะดึงเฟรมที่กระจายเท่า ๆ กันมาตรวจ ตามจำนวน "Frames from video"
3. ปรับ confidence ของ rider และหมวก หรือเปิด **Contrast Enhancement**
4. ดูผลได้ 3 ส่วน
   - **Summary:** จำนวน rider, สวมหมวก, ไม่สวมหมวก และ Compliance
   - **กราฟและตารางรายภาพ:** ดาวน์โหลดเป็น CSV ได้
   - **ภาพผลลัพธ์:** 🟩 สวมหมวก 🟥 ไม่สวมหมวก 🟨 หมวก
5. แท็บ **Model Evaluation** แสดงผลประเมินโมเดลบน Test set

## Method

```
Image / Video ─→ (Contrast Enhancement) ─→ YOLO11n ─→ rider + with helmet
                                                     ─→ จับคู่หมวกกับ rider ─→ Helmet / No Helmet ─→ สถิติ
```

- **Model:** `yolo_10epoch_best.pt` เป็น YOLO11n ที่ fine-tune 10 epochs จาก Roboflow dataset
  [capstone_rider_helmet_rtms](https://universe.roboflow.com/capstone-qodg4/capstone_rider_helmet_rtms/dataset/1)
  มี 3 คลาส คือ `motorcycle`, `rider`, `with helmet`
- **Image Processing:** Contrast Enhancement (`src/preprocessing.py`) ใช้เทคนิคเดียวกับตอนเทรน
- **Helmet / No Helmet:** โมเดลไม่มีคลาส "ไม่สวมหมวก" จึงตัดสินดังนี้ ถ้าจุดกึ่งกลางของกล่องหมวกอยู่ในครึ่งบนของกล่อง rider ถือว่า rider คนนั้นสวมหมวก (`src/detection.py`)

## Evaluation

ผลบน Test set ของ dataset จำนวน 299 ภาพ คำนวณด้วย `evaluate.py`

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| rider | 0.818 | 0.717 | 0.732 | 0.376 |
| with helmet | 0.612 | 0.496 | 0.510 | 0.182 |

วิธีคำนวณใหม่ ใช้โฟลเดอร์ test ของ dataset ฟอร์แมต YOLO ที่มี `images/` และ `labels/`

```bash
python evaluate.py --test path/to/test
```

ผลจะถูกบันทึกใน `results/` ได้แก่ `metrics.json`, confusion matrix, PR curve และ F1 curve

## Limitations

- โมเดลหาหมวกเจอประมาณครึ่งเดียว (with helmet recall 0.50) คนที่สวมหมวกจึงอาจถูกตัดสินเป็น No Helmet
- Dataset แทบไม่มีผู้ขับขี่ที่ไม่สวมหมวก มีเพียง 9 จาก 6,171 ภาพใน train และ 0 ภาพใน valid/test

## Project Structure

```
├── app.py                 Streamlit web app
├── evaluate.py            ประเมินโมเดลบน Test set → results/
├── src/
│   ├── detection.py       detect, จับคู่หมวกกับ rider, วาดกรอบ, ดึงเฟรมวิดีโอ
│   └── preprocessing.py   Contrast Enhancement
├── yolo_10epoch_best.pt   โมเดลหลัก
├── images/test/           ภาพ/วิดีโอสำหรับทดสอบ (ที่มา: images/test/SOURCES.txt)
└── results/               ผลประเมิน
```
