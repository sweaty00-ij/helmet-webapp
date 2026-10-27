# Smart Helmet Detection

Web application ตรวจว่าผู้ขับขี่มอเตอร์ไซค์สวมหมวกกันน็อกหรือไม่ จากภาพหรือวิดีโอ
ใช้ YOLO11 + OpenCV + Streamlit (วิชา Image Processing and Computer Vision)

**Demo:** https://helmet-webapp.streamlit.app

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

1. อัปโหลดภาพหรือวิดีโอในช่องเดียวกัน ได้หลายไฟล์ ระบบแยกประเภทจากนามสกุลไฟล์
   - ภาพ: `.jpg .jpeg .png .bmp .webp`
   - วิดีโอ: `.mp4 .avi .mov .mkv .m4v .wmv` ดึงออกมา 5 เฟรมที่กระจายเท่า ๆ กันตลอดวิดีโอ
   - ลบไฟล์ทีละไฟล์ด้วยปุ่ม ✕ หรือลบทั้งหมดด้วยปุ่ม **🗑️ Clear all**
2. ทุกภาพถูกตรวจ 3 แบบ คือภาพต้นฉบับ, ภาพที่ลด contrast เหลือ ×0.3 และภาพจางนั้นหลังทำ **Histogram Equalization** แล้วแสดงผลเทียบกัน
3. ดูผลได้ 4 ส่วน
   - **Summary:** จำนวน rider, สวมหมวก, ไม่สวมหมวก และ Compliance ของภาพต้นฉบับ
   - **Histogram Equalization: before vs after:** ตารางเทียบผลรวมของทั้งสามแบบ
   - **กราฟและตารางรายภาพ:** ดาวน์โหลดเป็น CSV ได้
   - **ภาพผลลัพธ์:** 🟩 สวมหมวก 🟥 ไม่สวมหมวก 🟨 หมวก

## Method

```
Image / Video ─→ YOLO11n ─→ rider + with helmet
                         ─→ จับคู่หมวกกับ rider ─→ Helmet / No Helmet ─→ สถิติ
```

- **Model:** `yolo_10epoch_best.pt` เป็น YOLO11n ที่ fine-tune 10 epochs จาก Roboflow dataset
  [capstone_rider_helmet_rtms](https://universe.roboflow.com/capstone-qodg4/capstone_rider_helmet_rtms/dataset/1)
  มี 3 คลาส คือ `motorcycle`, `rider`, `with helmet`
- **Image Processing:** จำลองภาพคุณภาพแย่ (เช่น หมอก แสงน้อย) ด้วยการลด contrast เหลือ ×0.3
  แล้วกู้คืนด้วย Histogram Equalization (`cv2.equalizeHist`) เฉพาะช่องความสว่าง Y ของ YCrCb สีจึงไม่เพี้ยน (`app.py`)
  ผลบนภาพใน `images/test/` 14 ภาพ: ภาพต้นฉบับตรวจเจอ rider 15 คน, ภาพจาง ×0.3 เหลือ 8 คน
  และหลัง Histogram Equalization กลับมาเจอ 15 คน ผลตรวจหลักของเว็บ (Summary) ใช้ภาพต้นฉบับ
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
│   └── preprocessing.py   Contrast Enhancement จาก Week 1 (เวอร์ชันก่อนหน้า เว็บไม่ได้เรียกใช้แล้ว)
├── yolo_10epoch_best.pt   โมเดลหลัก
├── images/test/           ภาพ/วิดีโอสำหรับทดสอบ (ที่มา: images/test/SOURCES.txt)
└── results/               ผลประเมิน
```
