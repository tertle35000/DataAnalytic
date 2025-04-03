import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

font_path = "C:/Windows/Fonts/THSarabun.ttf"  # ใช้ฟอนต์ที่เจอจากคำสั่งก่อนหน้า
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# Set Sheet ID and Sheet Name
sheet_id = "1YX3HKrvKoYYQeSielEAUXf0DWAPm_by-BJP2lBpxpys"
sheet_name = "Dataproject"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# Load data into Pandas
df = pd.read_csv(url)

# แปลงค่าข้อความในคอลัมน์ 'ช่วงเวลาที่มักเสพสื่อมากที่สุด :' และ 'อายุ :' ให้เป็นค่าหมายเลข
le_time = LabelEncoder()
df['ช่วงเวลาที่มักเสพสื่อมากที่สุด :'] = le_time.fit_transform(df['ช่วงเวลาที่มักเสพสื่อมากที่สุด :'])

le_age = LabelEncoder()
df['อายุ : '] = le_age.fit_transform(df['อายุ : '])

# สมมติว่า 'ช่วงเวลาที่มักเสพสื่อมากที่สุด :' คือคุณสมบัติ และ 'อายุ :' คือค่าที่ต้องการทำนาย
X = df[['ช่วงเวลาที่มักเสพสื่อมากที่สุด :']]  # คุณสมบัติ
y = df['อายุ : ']  # เป้าหมาย

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# วัดความแม่นยำ
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# พล็อตกราฟเปรียบเทียบค่าจริงและค่าทำนาย
plt.figure(figsize=(10, 6))

# แสดงค่าจริงและค่าทำนายในกราฟแท่ง
x_axis = np.arange(len(y_test))  # สร้างแกน x สำหรับการพล็อตกราฟ
plt.bar(x_axis - 0.2, y_test, width=0.4, label='Actual', alpha=0.6, color='blue')
plt.bar(x_axis + 0.2, y_pred, width=0.4, label='Predicted', alpha=0.6, color='orange')

# เพิ่มรายละเอียดกราฟ
plt.title("การเปรียบเทียบค่าจริงและค่าทำนาย (Random Forest)", fontproperties=font_prop)
plt.xlabel("ตัวอย่างข้อมูล", fontproperties=font_prop)
plt.ylabel("ช่วงอายุ", fontproperties=font_prop)
plt.legend()

# แสดงกราฟ
plt.tight_layout()
plt.show()
